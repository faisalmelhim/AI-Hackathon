#
# modeling.py: FastAPI router for financial modeling.
# - POST endpoint to run a Discounted Cash Flow (DCF) valuation.
#
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import the specific Pydantic models used in this router
from app.models.schemas import DCFRequest, DCFResponse, DCFYearlyProjection

router = APIRouter()

def _calculate_dcf_scenario(params: DCFRequest) -> tuple[float, List[DCFYearlyProjection]]: # <--- FIX: Changed type hint syntax
    """Core DCF calculation logic."""
    if params.discount_rate <= params.terminal_growth:
        raise ValueError("Discount rate must be greater than terminal growth rate.")

    yearly_projections = []
    revenues = [params.current_revenue]
    
    # Project 5 years of financials
    for i in range(5):
        revenues.append(revenues[-1] * (1 + params.growth_rates[i]))
    
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
        yearly_projections.append(DCFYearlyProjection(year=i, revenue=revenue, ebit=ebit, fcf=fcf))
        
    fcf_year5 = free_cash_flows[-1]
    terminal_value = (fcf_year5 * (1 + params.terminal_growth)) / (params.discount_rate - params.terminal_growth)
    
    dcf_values = [fcf / ((1 + params.discount_rate) ** (i + 1)) for i, fcf in enumerate(free_cash_flows)]
    discounted_tv = terminal_value / ((1 + params.discount_rate) ** 5)
    
    return sum(dcf_values) + discounted_tv, yearly_projections

@router.post("/model/dcf", response_model=DCFResponse)
def run_dcf_model(request: DCFRequest = Body(...)):
    """
    Performs a 5-year DCF valuation with base, bull, and bear scenarios.
    """
    try:
        base_npv, yearly_data = _calculate_dcf_scenario(request)
        
        bull_params = request.model_copy()
        bull_params.growth_rates = [min(g + 0.03, 0.95) for g in request.growth_rates]
        bull_params.operating_margin = min(request.operating_margin + 0.02, 0.95)
        bull_npv, _ = _calculate_dcf_scenario(bull_params)

        bear_params = request.model_copy()
        bear_params.growth_rates = [max(g - 0.03, -0.95) for g in request.growth_rates]
        bear_params.operating_margin = max(request.operating_margin - 0.02, -0.95)
        bear_npv, _ = _calculate_dcf_scenario(bear_params)

        return DCFResponse(
            base=base_npv,
            bull=bull_npv,
            bear=bear_npv,
            yearly=yearly_data,
            assumptions_used=request.model_dump()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))