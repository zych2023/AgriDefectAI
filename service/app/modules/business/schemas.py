from datetime import datetime
from pydantic import BaseModel, Field


# --- Field ---
class FieldCreate(BaseModel):
    field_name: str | None = None
    area: float | None = None
    location: str | None = None
    soil_type: str | None = None
    current_crop_id: int | None = None
    status: str = "fallow"
    remarks: str | None = None


class FieldUpdate(BaseModel):
    field_name: str | None = None
    area: float | None = None
    location: str | None = None
    soil_type: str | None = None
    current_crop_id: int | None = None
    status: str | None = None
    has_pest_disease: int | None = None
    remarks: str | None = None


class FieldResponse(BaseModel):
    id: int
    farmer_id: int
    field_name: str | None = None
    area: float | None = None
    location: str | None = None
    soil_type: str | None = None
    current_crop_id: int | None = None
    status: str
    has_pest_disease: int
    remarks: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Crop (reference table) ---
class CropCreate(BaseModel):
    name: str = Field(..., max_length=50)
    variety: str | None = None
    growth_cycle: int | None = None
    description: str | None = None


class CropUpdate(BaseModel):
    name: str | None = None
    variety: str | None = None
    growth_cycle: int | None = None
    description: str | None = None


class CropResponse(BaseModel):
    id: int
    name: str
    variety: str | None = None
    growth_cycle: int | None = None
    description: str | None = None

    class Config:
        from_attributes = True


# --- Pest ---
class PestCreate(BaseModel):
    name: str = Field(..., max_length=100)
    type: str = Field(default="disease", pattern=r"^(disease|pest|weed)$")
    symptoms: str | None = None
    pathogen: str | None = None
    prevention: str | None = None
    example_image_url: str | None = None


class PestUpdate(BaseModel):
    name: str | None = None
    type: str | None = None
    symptoms: str | None = None
    pathogen: str | None = None
    prevention: str | None = None
    example_image_url: str | None = None


class PestResponse(BaseModel):
    id: int
    name: str
    type: str
    symptoms: str | None = None
    pathogen: str | None = None
    prevention: str | None = None
    example_image_url: str | None = None

    class Config:
        from_attributes = True


# --- FieldPestRecord ---
class FieldPestRecordResponse(BaseModel):
    id: int
    field_id: int
    pest_id: int
    detected_at: datetime
    severity: str
    notes: str | None = None

    class Config:
        from_attributes = True
