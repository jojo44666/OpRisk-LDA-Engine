import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class OpRiskVaRModel:
    def __init__(self, data: np.array, risk_type_name: str = "Internal Fraud"):
        self.data = data
        self.risk_type = risk_type_name
        self.lambda_freq = None
        self.gpd_params = None
        self.annual_losses = None
        self.var_999 = None
        self.expected_loss = None
        
        print(f"[{self.risk_type}] Model Initialized with {len(data)} historical events.")

    def fit_frequency(self, years_history: int) -> float:
        avg_events = len(self.data) / years_history
        self.lambda_freq = avg_events
        print(f" >> Frequency Fit (Poisson): λ = {self.lambda_freq:.4f} events/year")
        return self.lambda_freq

    def fit_severity(self, threshold_quantile: float = 0.0) -> tuple:
        threshold = np.percentile(self.data, threshold_quantile * 100)
        tail_data = self.data[self.data >= threshold]
        
        self.gpd_params = stats.genpareto.fit(tail_data)
        
        c, loc, scale = self.gpd_params
        print(f" >> Severity Fit (GPD): Shape(ξ)={c:.4f}, Loc={loc:.2f}, Scale={scale:.2f}")
        return self.gpd_params

    def monte_carlo_simulation(self, n_years: int = 1_000_000) -> np.array:
        if self.lambda_freq is None or self.gpd_params is None:
            raise ValueError("Fit frequency and severity models before running simulation.")

        print(f" >> Starting Monte Carlo Simulation ({n_years:,.0f} years)...")
        
        n_events_per_year = np.random.poisson(self.lambda_freq, n_years)
        
        total_events = np.sum(n_events_per_year)
        
        c, loc, scale = self.gpd_params
        all_severities = stats.genpareto.rvs(c, loc=loc, scale=scale, size=total_events)
        
        
        year_indices = np.repeat(np.arange(n_years), n_events_per_year)
        
        self.annual_losses = np.bincount(year_indices, weights=all_severities)
        
        if len(self.annual_losses) < n_years:
            padding = np.zeros(n_years - len(self.annual_losses))
            self.annual_losses = np.concatenate([self.annual_losses, padding])

        print(f" >> Simulation Complete. Generated distribution of {len(self.annual_losses)} years.")
        return self.annual_losses

    def calculate_capital(self) -> dict:
        if self.annual_losses is None:
            raise ValueError("Run simulation first.")
            
        self.var_999 = np.percentile(self.annual_losses, 99.9)
        self.expected_loss = np.mean(self.annual_losses)
        unexpected_loss = self.var_999 - self.expected_loss
        
        results = {
            "VaR_99.9": self.var_999,
            "Expected_Loss": self.expected_loss,
            "Unexpected_Loss (Capital)": unexpected_loss
        }
        
        print("-" * 40)
        print(f"CAPITAL CALCULATION RESULTS ({self.risk_type})")
        print("-" * 40)
        print(f"OpRisk VaR (99.9%):      ${self.var_999:,.2f}")
        print(f"Expected Annual Loss:    ${self.expected_loss:,.2f}")
        print(f"Regulatory Capital (UL): ${unexpected_loss:,.2f}")
        print("-" * 40)
        
        return results

    def plot_dashboard(self):
        fig = plt.figure(constrained_layout=True, figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(np.random.poisson(self.lambda_freq, 10000), bins=30, discrete=True, color='teal', ax=ax1)
        ax1.set_title(f"Frequency Distribution (Poisson λ={self.lambda_freq:.1f})")
        ax1.set_xlabel("Events per Year")

        ax2 = fig.add_subplot(gs[0, 1])
        tail_viz_data = self.data[self.data < np.percentile(self.data, 95)] 
        sns.histplot(tail_viz_data, bins=50, kde=True, color='orange', ax=ax2)
        ax2.set_title("Severity Distribution (Historical Data Body)")
        ax2.set_xlabel("Loss Amount ($)")

        ax3 = fig.add_subplot(gs[1, :])
        
        sns.histplot(self.annual_losses, bins=100, stat='density', color='navy', alpha=0.6, label='Simulated Annual Losses')
        sns.kdeplot(self.annual_losses, color='black', linewidth=1)
        
        ax3.axvline(self.var_999, color='red', linestyle='--', linewidth=2.5, label=f'VaR 99.9%: ${self.var_999:,.0f}')
        ax3.axvline(self.expected_loss, color='green', linestyle='--', linewidth=2.5, label=f'Expected Loss: ${self.expected_loss:,.0f}')
        
        ax3.text(self.var_999 * 1.05, ax3.get_ylim()[1]*0.8, 
                 f'Regulatory Capital\nRequired:\n${self.var_999:,.0f}', 
                 color='red', fontweight='bold')

        ax3.set_title(f"Annual Loss Distribution (1 Million Year Simulation) - {self.risk_type}", fontsize=14)
        ax3.set_xlabel("Total Annual Loss ($)")
        ax3.set_xlim(0, np.percentile(self.annual_losses, 99.95)) 
        ax3.legend()

        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    n_historical_events = 500
    
    body_losses = np.random.lognormal(mean=8, sigma=1.5, size=int(n_historical_events * 0.9))
    tail_losses = np.random.pareto(a=1.5, size=int(n_historical_events * 0.1)) * 50000 + 10000
    
    historical_losses = np.concatenate([body_losses, tail_losses])
    
    model = OpRiskVaRModel(data=historical_losses, risk_type_name="Cyber & Fraud Risk")
    
    model.fit_frequency(years_history=20)
    model.fit_severity(threshold_quantile=0.0) 
    
    model.monte_carlo_simulation(n_years=1_000_000) 
    
    model.calculate_capital()
    
    model.plot_dashboard()