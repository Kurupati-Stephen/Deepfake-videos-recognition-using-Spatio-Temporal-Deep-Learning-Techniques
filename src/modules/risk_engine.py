import re

class RiskEngine:
    def __init__(self):
        # Keywords mapped to specific Risk Types
        self.risk_mapping = {
            "Financial Fraud / Scam": ["money", "bank", "account", "transfer", "crypto", "investment", "urgent", "payment"],
            "Political / National Security Threat": ["election", "president", "vote", "government", "military", "war", "minister"],
            "Privacy Violation": ["leak", "private", "nude", "address", "phone number", "confidential"],
            "Cyberbullying / Harassment": ["hate", "bitch", "ugly", "kill", "die", "stupid", "idiot"],
            "Misinformation / Fake News": ["news", "breaking", "report", "anchor", "studio", "vaccine", "hoax"],
        }
        
    def classify_risk(self, prediction, context_text=""):
        if prediction == "AUTHENTIC":
            return "No Immediate Risk (Authentic)"
            
        context_lower = context_text.lower()
        
        # Check rule-based NLP keywords
        for risk_type, keywords in self.risk_mapping.items():
            for kw in keywords:
                if kw in context_lower:
                    return risk_type
                    
        # Default fallback
        return "Personal Reputation Damage / General Deepfake"

    def assign_threat_level(self, risk_type, prediction_prob):
        if prediction_prob <= 0.5:
            return "LOW"
            
        if risk_type in ["Financial Fraud / Scam", "Political / National Security Threat"]:
            return "CRITICAL"
        elif risk_type in ["Privacy Violation", "Cyberbullying / Harassment", "Personal Reputation Damage / General Deepfake"]:
            return "HIGH"
        elif risk_type in ["Misinformation / Fake News"]:
            return "MEDIUM"
        
        return "LOW"

    def get_recommendation(self, risk_type):
        recs = {
            "Financial Fraud / Scam": "Do not send money. Verify identity directly via trusted channels.",
            "Political / National Security Threat": "Critical: manual verification required. Escalate to authorities.",
            "Privacy Violation": "Report misuse immediately. Restrict sharing.",
            "Cyberbullying / Harassment": "Block, report, and preserve evidence offline.",
            "Misinformation / Fake News": "Verify source before sharing. Do NOT circulate.",
            "Personal Reputation Damage / General Deepfake": "Do not circulate. Save as evidence.",
            "No Immediate Risk (Authentic)": "Media is certified authentic. No action required."
        }
        return recs.get(risk_type, "Use report for forensic review.")
