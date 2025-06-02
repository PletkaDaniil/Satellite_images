from project.data import *

ee.Authenticate()
ee.Initialize(project='your-project-id')

region = ee.Geometry.Rectangle([
    30.17872139688425, 59.86816653184917,
    30.30690369635918, 59.9963488313241
])

tensor, meta_data = get_image_tensor(
    region=region,
    start_date="2018-01-01",
    end_date="2022-12-31",
    collection_id="LANDSAT/LC08/C02/T1_L2",
    bands=["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"],
    cloud_cover_max=40,
    scale=30,
    crs = "EPSG:4326",
    base_output_dir = 'data'
)

print(tensor.shape)

tensor, meta_data = get_image_tensor(
    region=region,
    start_date="2023-01-01",
    end_date="2024-12-31",
    collection_id="LANDSAT/LC09/C02/T1_L2",
    bands=["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"],
    cloud_cover_max=40,
    scale=30,
    crs = "EPSG:4326",
    base_output_dir = 'data',
    existing_tensor=tensor,
    existing_metadata=meta_data,
    flag=True
)

print(tensor.shape)