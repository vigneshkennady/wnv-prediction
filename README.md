<<<<<<< HEAD
# wnv-prediction
Predicting county level WNV cases
=======
# West Nile Virus (WNV) County-Level Prediction Project

This project acquires and processes mosquito surveillance and environmental data for West Nile Virus prediction across 6 US counties.

## Target Counties
- **Larimer County, CO** (FIPS: 08069)
- **Boulder County, CO** (FIPS: 08013) 
- **Dallas County, TX** (FIPS: 48113)
- **Maricopa County, AZ** (FIPS: 04013)
- **Cook County, IL** (FIPS: 17031)
- **Los Angeles County, CA** (FIPS: 06037)

## Data Sources

### Mosquito Surveillance Data (`MosquitoSurv.py`)
- **Cook County, IL**: Chicago Data Portal Socrata API
- **Los Angeles County, CA**: westnile.ca.gov annual CSV downloads
- **Maricopa County, AZ**: VectorSurv REST API (requires token)
- **Colorado Counties**: CDPHE HTML summary + manual files
- **Dallas County, TX**: DSHS weekly arbovirus PDF reports
- **All Counties**: CDC ArboNET annual backstop

### Land Use/Crop Data (`landUseData.py`)
- **Cropland Data Layer (CDL)**: Multiple acquisition strategies
  - Strategy A: CropScape bbox API
  - Strategy B: Google Earth Engine (recommended)
  - Strategy C: NASS FTP direct download
  - Strategy D: Microsoft Planetary Computer STAC API

### Demographics Data (`demographicsData.py`)
- **American Community Survey**: Census Bureau demographic indicators

## Installation

```bash
# Clone repository
git clone <repository-url>
cd westNile

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Create `requirements.txt` with:
```
pandas>=1.5.0
requests>=2.25.0
geopandas>=0.12.0
pyproj>=3.3.0
tqdm>=4.64.0
beautifulsoup4>=4.10.0
lxml>=4.6.3
pdfplumber>=0.11.0
```

Optional dependencies:
```
# For Google Earth Engine (land use Strategy B)
earthengine-api

# For Microsoft Planetary Computer (land use Strategy D)
pystac-client
odc-stac
rioxarray
xarray
numpy

# For VectorSurv API (Maricopa County)
# Set environment variable after getting token from https://vectorsurv.org/signup
export VECTORSURV_TOKEN=your_token_here

# For Chicago Data Portal (optional, higher rate limits)
export CHICAGO_APP_TOKEN=your_token_here
```

## Usage

### Run Mosquito Surveillance Data Acquisition
```bash
python MosquitoSurv.py
```

### Run Land Use Data Acquisition
```bash
python landUseData.py
```

### Run Demographics Data Acquisition  
```bash
python demographicsData.py
```

## Output Structure

```
westNile/
├── output/                    # Generated data files
│   ├── mosquito_surveillance_county_week.csv
│   ├── mosquito_surveillance_pool_level.csv
│   ├── demographics_acs.csv
│   └── cdl_strategy_*.csv
├── manual_data/               # Manual data fallbacks
│   ├── co_cdphe/            # Colorado annual files
│   └── dallas_tx/            # Dallas County CSVs
└── src/                      # Source code
    ├── MosquitoSurv.py
    ├── landUseData.py
    └── demographicsData.py
```

## Data Model Features

### Mosquito Surveillance (county-week level)
- `county_key`: County identifier
- `fips`: 5-digit FIPS code  
- `year`: Surveillance year
- `mmwr_week`: MMWR epidemiological week
- `total_pools_tested`: Total pools tested
- `positive_pools`: WNV-positive pools
- `pct_positive`: % positive pools
- `total_mosquitoes`: Total mosquitoes
- `culex_pools`: Culex species pools
- `culex_pct`: % Culex pools
- `mir_per_1000`: Minimum infection rate
- `source`: Data source identifier

### Land Use (CDL)
- `cdl_rice_acres`: Rice cultivation area
- `cdl_corn_acres`: Corn cultivation area  
- `cdl_wetlands_acres`: Wetland area
- `cdl_developed_acres`: Urban/developed area
- And other crop-specific metrics

### Demographics
- `total_pop`: Total population
- `median_hh_income`: Median household income
- `pop_65_plus`: Population 65+ years
- `poverty_rate`: Poverty percentage
- `housing_units`: Total housing units

## Configuration

### Environment Variables
```bash
export VECTORSURV_TOKEN=your_vectorsurv_token
export CHICAGO_APP_TOKEN=your_chicago_token
```

### Manual Data Setup
1. **Colorado Data**: Download annual CDPHE files to `./manual_data/co_cdphe/`
2. **Dallas Data**: Place surveillance CSVs in `./manual_data/dallas_tx/`
3. **Maricopa Data**: Set `VECTORSURV_TOKEN` environment variable

## Troubleshooting

### Common Issues
1. **SSL Certificate Errors**: Some endpoints have expired certificates
2. **Rate Limiting**: Use API tokens for higher limits
3. **Missing Data**: Use manual data fallback directories
4. **PDF Parsing Errors**: Ensure `pdfplumber` is installed

### Data Source Status
Run pre-flight checks to verify endpoint availability:
```bash
python -c "from MosquitoSurv import run_preflight; run_preflight()"
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit changes: `git commit -m 'Description of changes'`
5. Push to fork: `git push origin feature-name`
6. Create Pull Request

## License

This project is for research and educational purposes. Please cite appropriately if used in academic work.

## Contact

For questions about data sources or methodology, please refer to the documentation within each script or create an issue in the repository.
>>>>>>> 2317864 (Initial commit: West Nile Virus prediction project with mosquito surveillance, land use, and demographics data acquisition scripts)
