import os
from pathlib import Path
from typing import Dict, Optional
import statistics

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv, BatchNorm


class ImprovedRespiratoryGAT(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                conv = GATConv(hidden_dim, max(1, hidden_dim // num_heads), heads=num_heads, concat=True, dropout=dropout)
            else:
                conv = SAGEConv(hidden_dim, hidden_dim)
            self.gat_layers.append(conv)
            self.norms.append(BatchNorm(hidden_dim))

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, max(4, hidden_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(4, hidden_dim // 4), 1),
        )
        self.wheeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.crackle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.log_var_wheeze = nn.Parameter(torch.tensor(0.0))
        self.log_var_crackle = nn.Parameter(torch.tensor(0.0))
        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else None

        x = x.to(self.input_proj[0].weight.device)
        edge_index = edge_index.to(self.input_proj[0].weight.device)

        x = self.input_proj(x)
        residuals = []
        for i, (conv, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x_new = conv(x, edge_index)
            x_new = F.elu(x_new)
            x_new = norm(x_new)
            if i > 0 and i % 2 == 0:
                x_new = x_new + residuals[-1]
            x = F.dropout(x_new, p=self.dropout, training=self.training)
            residuals.append(x)

        if batch is not None:
            batch = batch.to(x.device)
            attn_scores = self.attention_pool(x).squeeze(-1)
            x_graph = []
            for b in torch.unique(batch):
                mask = batch == b
                scores = attn_scores[mask]
                weights = torch.softmax(scores, dim=0).unsqueeze(-1)
                x_graph.append((x[mask] * weights).sum(dim=0))
            x = torch.stack(x_graph, dim=0)
        else:
            attn_scores = self.attention_pool(x).squeeze(-1)
            weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)
            x = (x * weights).sum(dim=0, keepdim=True)

        w_logits = self.wheeze_head(x).squeeze(-1)
        c_logits = self.crackle_head(x).squeeze(-1)
        return w_logits, c_logits


def load_gnn_model(checkpoint_path: str, device: torch.device) -> ImprovedRespiratoryGAT:
    model = ImprovedRespiratoryGAT(input_dim=768, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.4)
    model = model.to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Warning: GNN checkpoint '{checkpoint_path}' not found. Using randomly initialized model.")
    model.eval()
    return model


def build_chain_edge_index(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    edges = [[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def estimate_breathing_rate_bpm(y: np.ndarray, sr: int, audio_duration_s: float) -> Optional[float]:
    if len(y) < sr:
        return None
    frame_len = int(0.2 * sr)
    hop_len = int(0.05 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    if len(rms) == 0:
        return None
    win = 5
    smooth = np.convolve(rms, np.ones(win) / win, mode="same") if len(rms) >= win else rms
    thr = float(smooth.mean() + 0.5 * smooth.std())
    times = librosa.frames_to_time(np.arange(len(smooth)), sr=sr, hop_length=hop_len)
    min_interval = 0.8
    peaks = []
    last_t = -1e9
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > thr:
            t = float(times[i])
            if t - last_t >= min_interval:
                peaks.append(t)
                last_t = t
    if len(peaks) < 2:
        return None
    return float((len(peaks) / max(audio_duration_s, 1e-6)) * 60.0)


class PatientStateManager:
    def __init__(
        self,
        ema_alpha: float = 0.12,
        low_delta: float = 0.08,
        high_delta: float = 0.20,
        min_samples_for_baseline: int = 5,
        force_established_after_s: float = 10.0,
    ):
        self.ema_alpha = ema_alpha
        self.low_delta = low_delta
        self.high_delta = high_delta
        self.min_samples_for_baseline = min_samples_for_baseline
        self.force_established_after_s = force_established_after_s
        self.patient_data: Dict[str, Dict] = {}

    def update_and_get_state(
        self,
        patient_id: str,
        wheeze_prob: float,
        crackle_prob: float,
        timestamp: float = 0.0,
    ) -> Dict:
        if patient_id not in self.patient_data:
            self.patient_data[patient_id] = {
                "wheeze_ema": wheeze_prob,
                "crackle_ema": crackle_prob,
                "wheeze_baseline": None,
                "crackle_baseline": None,
                "wheeze_history": [],
                "crackle_history": [],
                "timestamps": [],
                "count": 0,
                "baseline_established": False,
                "breathing_rate_history": [],
            }

        data = self.patient_data[patient_id]
        data["timestamps"].append(timestamp)
        data["wheeze_history"].append(wheeze_prob)
        data["crackle_history"].append(crackle_prob)
        data["count"] += 1

        if data["count"] == 1:
            data["wheeze_ema"] = wheeze_prob
            data["crackle_ema"] = crackle_prob
        else:
            data["wheeze_ema"] = self.ema_alpha * wheeze_prob + (1 - self.ema_alpha) * data["wheeze_ema"]
            data["crackle_ema"] = self.ema_alpha * crackle_prob + (1 - self.ema_alpha) * data["crackle_ema"]

        if data["count"] >= self.min_samples_for_baseline and not data["baseline_established"]:
            data["wheeze_baseline"] = float(np.mean(data["wheeze_history"][-self.min_samples_for_baseline :]))
            data["crackle_baseline"] = float(np.mean(data["crackle_history"][-self.min_samples_for_baseline :]))
            data["baseline_established"] = True

        result = {"overall_state": "establishing", "reason": {}}
        for axis, ema, baseline, history in [
            ("wheeze", data["wheeze_ema"], data["wheeze_baseline"], data["wheeze_history"]),
            ("crackle", data["crackle_ema"], data["crackle_baseline"], data["crackle_history"]),
        ]:
            if not data["baseline_established"] or baseline is None:
                state = "establishing"
                delta = 0.0
                trend = 0.0
            else:
                delta = float(ema - baseline)
                trend = float(history[-1] - history[-3]) if len(history) >= 3 else 0.0
                if abs(delta) < self.low_delta:
                    state = "green"
                elif abs(delta) < self.high_delta:
                    state = "orange"
                else:
                    state = "red"

            result["reason"][axis] = {
                "baseline": baseline,
                "value": ema,
                "delta": delta,
                "trend": trend,
                "state": state,
            }

        if timestamp >= self.force_established_after_s and not data["baseline_established"] and data["count"] > 0:
            data["baseline_established"] = True
            data["wheeze_baseline"] = float(np.mean(data["wheeze_history"]))
            data["crackle_baseline"] = float(np.mean(data["crackle_history"]))
            for axis in ["wheeze", "crackle"]:
                result["reason"][axis]["baseline"] = data[f"{axis}_baseline"]
                result["reason"][axis]["state"] = "green"

        for s in ["red", "orange", "green", "establishing"]:
            if s in [result["reason"][axis]["state"] for axis in ["wheeze", "crackle"]]:
                result["overall_state"] = s
                break

        result["count"] = data["count"]
        return result


class ClinicalReferenceRanges:
    RESPIRATORY_RATES = {
        (0, 1): (30, 60),
        (1, 2): (24, 40),
        (2, 6): (22, 34),
        (6, 12): (18, 30),
        (12, 18): (12, 20),
        (18, 65): (12, 20),
        (65, 150): (12, 28),
    }
    SEX_ADJUSTMENT = {"male": 0.0, "female": 2.0}
    RESPIRATORY_SEVERITY = {
        "bradypnea": {"threshold": 8, "severity": "red"},
        "low_normal": {"threshold": 12, "severity": "green"},
        "high_normal": {"threshold": 20, "severity": "green"},
        "tachypnea_mild": {"threshold": 24, "severity": "orange"},
        "tachypnea_moderate": {"threshold": 28, "severity": "orange"},
        "tachypnea_severe": {"threshold": 30, "severity": "red"},
    }
    WHEEZE_THRESHOLDS = {
        "normal": 0.30,
        "borderline": 0.45,
        "abnormal": 0.60,
        "severe": 0.75,
    }
    CRACKLE_THRESHOLDS = {
        "normal": 0.35,
        "borderline": 0.50,
        "abnormal": 0.65,
        "severe": 0.80,
    }

    @classmethod
    def get_normal_breathing_range(cls, age_years: float, sex: str = "male"):
        for (min_age, max_age), (low, high) in cls.RESPIRATORY_RATES.items():
            if min_age <= age_years <= max_age:
                if age_years >= 18:
                    adj = cls.SEX_ADJUSTMENT.get(str(sex).lower(), 0.0)
                    return (low + adj, high + adj)
                return (low, high)
        return (12, 20)

    @classmethod
    def assess_respiratory_rate(cls, rate_bpm: float, age_years: float, sex: str = "male"):
        normal_low, normal_high = cls.get_normal_breathing_range(age_years, sex)
        if rate_bpm < cls.RESPIRATORY_SEVERITY["bradypnea"]["threshold"]:
            severity = "red"
            status = "Severe bradypnea (dangerously low breathing rate)"
            clinical_action = "Immediate clinical review required"
        elif rate_bpm < normal_low:
            severity = "orange"
            status = f"Mild bradypnea (below normal range {normal_low}-{normal_high})"
            clinical_action = "Monitor closely, consider clinical assessment"
        elif rate_bpm <= normal_high:
            severity = "green"
            status = f"Normal respiratory rate ({normal_low}-{normal_high} bpm for age/sex)"
            clinical_action = "Routine monitoring"
        elif rate_bpm < cls.RESPIRATORY_SEVERITY["tachypnea_mild"]["threshold"]:
            severity = "orange"
            status = f"Mild tachypnea (above normal range {normal_low}-{normal_high})"
            clinical_action = "Monitor, assess for underlying cause"
        elif rate_bpm < cls.RESPIRATORY_SEVERITY["tachypnea_moderate"]["threshold"]:
            severity = "orange"
            status = "Moderate tachypnea"
            clinical_action = "Clinical assessment recommended"
        else:
            severity = "red"
            status = "Severe tachypnea (significant respiratory distress)"
            clinical_action = "Urgent clinical review required"

        return {
            "value": float(rate_bpm),
            "normal_range": [float(normal_low), float(normal_high)],
            "status": status,
            "severity": severity,
            "clinical_action": clinical_action,
        }

    @classmethod
    def assess_adventitious_sounds(cls, wheeze_prob: float, crackle_prob: float):
        if wheeze_prob < cls.WHEEZE_THRESHOLDS["normal"]:
            wheeze_severity = "green"
            wheeze_status = "Normal - no significant wheeze detected"
        elif wheeze_prob < cls.WHEEZE_THRESHOLDS["borderline"]:
            wheeze_severity = "green"
            wheeze_status = "Borderline - subtle wheeze, monitor"
        elif wheeze_prob < cls.WHEEZE_THRESHOLDS["abnormal"]:
            wheeze_severity = "orange"
            wheeze_status = "Abnormal - clinically significant wheeze"
        else:
            wheeze_severity = "red"
            wheeze_status = "Severe - prominent wheeze, indicates airway obstruction"

        if crackle_prob < cls.CRACKLE_THRESHOLDS["normal"]:
            crackle_severity = "green"
            crackle_status = "Normal - no significant crackles detected"
        elif crackle_prob < cls.CRACKLE_THRESHOLDS["borderline"]:
            crackle_severity = "green"
            crackle_status = "Borderline - fine crackles, monitor"
        elif crackle_prob < cls.CRACKLE_THRESHOLDS["abnormal"]:
            crackle_severity = "orange"
            crackle_status = "Abnormal - clinically significant crackles"
        else:
            crackle_severity = "red"
            crackle_status = "Severe - prominent crackles, indicates interstitial pathology"

        severity_rank = {"green": 0, "orange": 1, "red": 2}
        overall_severity = "green"
        for s in (wheeze_severity, crackle_severity):
            if severity_rank.get(s, 0) > severity_rank.get(overall_severity, 0):
                overall_severity = s

        return {
            "wheeze": {
                "probability": float(wheeze_prob),
                "severity": wheeze_severity,
                "status": wheeze_status,
            },
            "crackle": {
                "probability": float(crackle_prob),
                "severity": crackle_severity,
                "status": crackle_status,
            },
            "overall_severity": overall_severity,
        }


class EnhancedPatientStateManager(PatientStateManager):
    def __init__(
        self,
        ema_alpha: float = 0.12,
        low_delta: float = 0.08,
        high_delta: float = 0.20,
        min_samples_for_baseline: int = 5,
        force_established_after_s: float = 10.0,
        patient_age: Optional[float] = None,
        patient_sex: Optional[str] = None,
    ):
        super().__init__(ema_alpha, low_delta, high_delta, min_samples_for_baseline, force_established_after_s)
        self.patient_age = patient_age
        self.patient_sex = patient_sex
        self.clinical_kb = ClinicalReferenceRanges()

    def set_patient_demographics(self, patient_id: str, age_years: float, sex: str) -> None:
        if patient_id not in self.patient_data:
            self.patient_data[patient_id] = {
                "wheeze_ema": 0.0,
                "crackle_ema": 0.0,
                "wheeze_baseline": None,
                "crackle_baseline": None,
                "wheeze_history": [],
                "crackle_history": [],
                "timestamps": [],
                "count": 0,
                "baseline_established": False,
            }
        self.patient_data[patient_id]["age"] = float(age_years)
        self.patient_data[patient_id]["sex"] = str(sex).lower()

    def update_and_get_clinical_state(
        self,
        patient_id: str,
        wheeze_prob: float,
        crackle_prob: float,
        breathing_rate: Optional[float] = None,
        timestamp: float = 0.0,
    ) -> Dict:
        base_state = self.update_and_get_state(patient_id, wheeze_prob, crackle_prob, timestamp)
        clinical = {}

        sound_assessment = self.clinical_kb.assess_adventitious_sounds(wheeze_prob, crackle_prob)
        clinical["adventitious_sounds"] = sound_assessment

        if breathing_rate is not None and breathing_rate > 0:
            pdata = self.patient_data.get(patient_id, {})
            if "breathing_rate_history" not in pdata:
                pdata["breathing_rate_history"] = []
            pdata["breathing_rate_history"].append(breathing_rate)

            age = pdata.get("age", self.patient_age if self.patient_age is not None else 40)
            sex = pdata.get("sex", self.patient_sex if self.patient_sex is not None else "male")
            rr_assessment = self.clinical_kb.assess_respiratory_rate(breathing_rate, age, sex)
            clinical["respiratory_rate"] = rr_assessment

        pdata = self.patient_data.get(patient_id, {})
        if pdata.get("breathing_rate_history"):
            base_state["breathing_rate_mean"] = float(statistics.mean(pdata["breathing_rate_history"]))
        else:
            base_state["breathing_rate_mean"] = None

        severity_rank = {"green": 0, "orange": 1, "red": 2}
        overall_sound = clinical.get("adventitious_sounds", {}).get("overall_severity", "green")
        max_severity = max(severity_rank.get(overall_sound, 0), severity_rank.get(clinical.get("respiratory_rate", {}).get("severity", "green"), 0))
        clinical["overall_clinical_status"] = {0: "green", 1: "orange", 2: "red"}.get(max_severity, "green")

        summary_parts = []
        if "adventitious_sounds" in clinical:
            ws = clinical["adventitious_sounds"]["wheeze"]["severity"]
            cs = clinical["adventitious_sounds"]["crackle"]["severity"]
            if ws != "green" or cs != "green":
                summary_parts.append(f"Adventitious sounds: wheeze={ws}, crackle={cs}")
        if "respiratory_rate" in clinical:
            rr = clinical["respiratory_rate"]
            summary_parts.append(f"Respiratory rate: {rr['value']:.1f} bpm ({rr['status']})")
        clinical["clinical_summary"] = " | ".join(summary_parts) if summary_parts else "No significant abnormalities detected"

        base_state["clinical_assessment"] = clinical
        base_state["comment"] = clinical.get("clinical_summary", "Establishing baseline...")
        return base_state


class ClinicalAlertSystem:
    @staticmethod
    def generate_alerts(clinical_state: Dict) -> list:
        alerts = []
        rr_assessment = clinical_state.get("clinical_assessment", {}).get("respiratory_rate", {})
        if rr_assessment:
            severity = rr_assessment.get("severity", "green")
            if severity == "red":
                alerts.append({
                    "priority": 1,
                    "type": "CRITICAL",
                    "message": rr_assessment.get("clinical_action", ""),
                    "detail": f"Respiratory rate {rr_assessment.get('value', 0.0):.1f} bpm - {rr_assessment.get('status', '')}",
                })
            elif severity == "orange":
                alerts.append({
                    "priority": 2,
                    "type": "WARNING",
                    "message": rr_assessment.get("clinical_action", ""),
                    "detail": f"Respiratory rate {rr_assessment.get('value', 0.0):.1f} bpm",
                })

        sound = clinical_state.get("clinical_assessment", {}).get("adventitious_sounds", {})
        for sound_type in ["wheeze", "crackle"]:
            s = sound.get(sound_type, {})
            severity = s.get("severity", "green")
            if severity == "red":
                alerts.append({
                    "priority": 1,
                    "type": "CRITICAL",
                    "message": f"Severe {sound_type} detected - clinical review required",
                    "detail": s.get("status", ""),
                })
            elif severity == "orange":
                alerts.append({
                    "priority": 2,
                    "type": "WARNING",
                    "message": f"Clinically significant {sound_type} detected",
                    "detail": s.get("status", ""),
                })

        reason = clinical_state.get("reason", {})
        for axis in ["wheeze", "crackle"]:
            ax = reason.get(axis, {})
            if ax.get("state") == "red":
                alerts.append({
                    "priority": 1,
                    "type": "CRITICAL",
                    "message": f"{axis.capitalize()} probability significantly elevated from baseline",
                    "detail": f"Delta: {ax.get('delta', 0.0):+.3f}, Trend: {ax.get('trend', 0.0):+.3f}",
                })
            elif ax.get("state") == "orange":
                alerts.append({
                    "priority": 2,
                    "type": "WARNING",
                    "message": f"{axis.capitalize()} probability moderately elevated",
                    "detail": f"Delta: {ax.get('delta', 0.0):+.3f}",
                })

        alerts.sort(key=lambda x: x["priority"])
        return alerts

    @staticmethod
    def get_triage_recommendation(alerts: list) -> Dict:
        if any(a["priority"] == 1 for a in alerts):
            return {
                "level": "EMERGENCY",
                "action": "Immediate clinical evaluation required",
                "timeframe": "Within 30 minutes",
                "setting": "Emergency Department",
            }
        if any(a["priority"] == 2 for a in alerts):
            return {
                "level": "URGENT",
                "action": "Clinical assessment recommended",
                "timeframe": "Within 24 hours",
                "setting": "Urgent Care / Primary Care",
            }
        if alerts:
            return {
                "level": "ROUTINE",
                "action": "Monitor per standard protocol",
                "timeframe": "As scheduled",
                "setting": "Primary Care / Home monitoring",
            }
        return {
            "level": "NORMAL",
            "action": "Continue routine monitoring",
            "timeframe": "Per clinical protocol",
            "setting": "Home / Primary Care",
        }


def create_clinical_report(
    patient_id: str,
    age: float,
    sex: str,
    wheeze_prob: float,
    crackle_prob: float,
    breathing_rate: Optional[float] = None,
) -> Dict:
    manager = EnhancedPatientStateManager()
    manager.set_patient_demographics(patient_id, age, sex)
    state = manager.update_and_get_clinical_state(patient_id, wheeze_prob, crackle_prob, breathing_rate)
    alerts = ClinicalAlertSystem.generate_alerts(state)
    triage = ClinicalAlertSystem.get_triage_recommendation(alerts)
    return {
        "patient_id": patient_id,
        "demographics": {"age": age, "sex": sex},
        "clinical_state": state,
        "alerts": alerts,
        "triage_recommendation": triage,
    }


def add_clinical_reasoning_to_output(output_dict: Dict, patient_age: Optional[float] = None, patient_sex: Optional[str] = None) -> Dict:
    result = output_dict.get("result", {})
    wheeze_prob = result.get("wheeze", {}).get("probability")
    crackle_prob = result.get("crackle", {}).get("probability")
    audio_id = result.get("audio_id")

    if wheeze_prob is None or crackle_prob is None or audio_id is None:
        return output_dict

    if patient_age is None:
        patient_age = 40
    if patient_sex is None:
        patient_sex = "male"

    clinical_report = create_clinical_report(
        patient_id=audio_id,
        age=patient_age,
        sex=patient_sex,
        wheeze_prob=wheeze_prob,
        crackle_prob=crackle_prob,
        breathing_rate=result.get("breathing_rate_bpm"),
    )
    output_dict["clinical_report"] = clinical_report
    return output_dict
