#
# modeling.py: FastAPI router for financial modeling.
# - POST endpoint to run a Discounted Cash Flow (DCF) valuation.
#
import numpy as np
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Router Setup ---
router = APIRouter()

# --- Pydantic Models ---
class DCFRequest(BaseModel):
    current_revenue: float = Field(..., gt=0, description="Most recent full-year revenue.")
    growth_rates: List[float] = Field(..., min_length=5, max_length=5, description="List of 5 projected annual revenue growth rates.")
    operating_margin: float = Field(..., gt=-1, lt=1, description="Projected stable operating margin.")
    tax_rate: float = Field(0.25, ge=0, lt=1, description="Effective tax rate.")
    capex_percent: float = Field(0.05, ge=0, lt=1, description="Capital expenditures as a percentage of revenue.")
    nwc_percent: float = Field(0.10, ge=0, lt=1, description="Net working capital as a percentage of the *change* in revenue.")
    discount_rate: float = Field(0.12, gt=0, lt=1, description="Weighted Average Cost of Capital (WACC).")
    terminal_growth: float = Field(0.03, ge=0, lt=1, description="Perpetual growth rate for terminal value.")

class DCFYearlyProjection(BaseModel):
    year: int
    revenue: float
    ebit: float
    fcf: float

class DCFResponse(BaseModel):
    base: float
    bull: float
    bear: float
    yearly: List[DCFYearlyProjection]
    assumptions_used: Dict[str, Any]


def calculate_dcf(params: DCFRequest) -> (float, List[Dict]):
    """Core DCF calculation logic."""
    if params.discount_rate <= params.terminal_growth:
        raise ValueError("Discount rate must be greater than terminal growth rate.")

    yearly_projections = []
    revenues = [params.current_revenue]
    
    # Project 5 years of financials
    for i in range(5):
        new_revenue = revenues[-1] * (1 + params.growth_rates[i])
        revenues.append(new_revenue)

    # Calculate FCF for each of the 5 projection years
    free_cash_flows = []
    for i in range(1, 6): # Years 1 to 5
        revenue = revenues[i]
        delta_revenue = revenue - revenues[i-1]
        
        ebit = revenue * params.operating_margin
        nopat = ebit * (1 - params.tax_rate)
        capex = revenue * params.capex_percent
        delta_nwc = delta_revenue * params.nwc_percent
        
        fcf = nopat - capex - delta_nwc
        free_cash_flows.append(fcf)
        yearly_projections.append({"year": i, "revenue": revenue, "ebit": ebit, "fcf": fcf})

    # Calculate Terminal Value using Gordon Growth Model
    fcf_year5 = free_cash_flows[-1]
    terminal_value = (fcf_year5 * (1 + params.terminal_growth)) / (params.discount_rate - params.terminal_growth)

    # Discount all cash flows to present value
    dcf_values = [fcf / ((1 + params.discount_rate) ** (i + 1)) for i, fcf in enumerate(free_cash_flows)]

    # Discount terminal value
    discounted_tv = terminal_value / ((1 + params.discount_rate) ** 5)
    
    total_npv = sum(dcf_values) + discounted_tv
    return total_npv, yearly_projections


@router.post("/api/model/dcf", response_model=DCFResponse)
def run_dcf_model(request: DCFRequest = Body(...)):
    """
    Performs a 5-year DCF valuation with base, bull, and bear scenarios.
    """
    try:
        # Base Case
        base_npv, yearly_data = calculate_dcf(request)
        
        # Bull Case
        bull_params = request.model_copy()
        bull_params.growth_rates = [min(g + 0.03, 0.95) for g in request.growth_rates] # Clamp growth
        bull_params.operating_margin = min(request.operating_margin + 0.02, 0.95) # Clamp margin
        bull_npv, _ = calculate_dcf(bull_params)

        # Bear Case
        bear_params = request.model_copy()
        bear_params.growth_rates = [max(g - 0.03, -0.50) for g in request.growth_rates] # Clamp growth
        bear_params.operating_margin = max(request.operating_margin - 0.02, -0.50) # Clamp margin
        bear_npv, _ = calculate_dcf(bear_params)

        return DCFResponse(
            base=base_npv,
            bull=bull_npv,
            bear=bear_npv,
            yearly=yearly_data,
            assumptions_used=request.model_dump()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred during DCF calculation: {e}")