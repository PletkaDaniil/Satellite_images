import os
import ee
import requests
import zipfile
import rioxarray
import numpy as np
from glob import glob
from io import BytesIO
from datetime import datetime
from sklearn.preprocessing import StandardScaler

ee.Authenticate()
ee.Initialize()

def normalize_tensor(tensor: np.ndarray):
    shape = tensor.shape
    reshaped_tensor = tensor.reshape(-1, shape[-1])

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(reshaped_tensor)
    
    return normalized_data.reshape(shape) 

def get_image_tensor(
    region, 
    start_date: str, 
    end_date: str, 
    collection_id: str,
    bands: list = None,
    cloud_cover_max: float = 40,
    cloud_cover_field: str = "CLOUD_COVER",
    scale: int = 30,
    crs: str = 'EPSG:4326',
    base_output_dir: str = 'data',
    existing_tensor: np.ndarray = None,
    existing_metadata: list = None,
    flag: bool = False
):
    os.makedirs(base_output_dir, exist_ok=True)

    col = (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt(cloud_cover_field, cloud_cover_max))
        .sort(cloud_cover_field)
    )
    
    count = col.size().getInfo()
    print(f"Найдено {count} снимков")

    images = col.toList(count)
    layers = []
    meta_data = []
    
    for ind in range(count):
        img = ee.Image(images.get(ind))
        date_str = img.date().format("YYYY-MM-dd").getInfo()
        month = datetime.strptime(date_str, "%Y-%m-%d").month
        if month in [12, 1, 2]:
            print(f"  [{ind+1}/{count}] Пропущено по соответствующим месяцам: {date_str}")
            continue

        image_id = img.id().getInfo()
        image_id_safe = image_id.replace('/', '_')
        zip_path = os.path.join(base_output_dir, f"{image_id_safe}.zip")
        extract_dir = os.path.join(base_output_dir, image_id_safe)

        if os.path.exists(extract_dir) and len(glob(f"{extract_dir}/*.tif")) > 0:
            print(f"  [{ind+1}/{count}] Ранее загружено: {image_id} ({date_str})")
        else:
            print(f"  [{ind+1}/{count}] Загрузка: {image_id} ({date_str})")

            if bands is not None:
                img = img.select(bands)
            else:
                bands = img.bandNames().getInfo()

            params = {
                'region': region.getInfo()['coordinates'],
                'scale': scale,
                'crs': crs,
                'fileFormat': 'GeoTIFF'
            }
            url = img.getDownloadURL(params)
            resp = requests.get(url)
            zip_buf = BytesIO(resp.content)

            with open(zip_path, 'wb') as f:
                f.write(zip_buf.getvalue())

            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)

            os.remove(zip_path)

        tif_files = sorted(glob(f"{extract_dir}/*.tif"))
        stack = []
        for tfile in tif_files:
            ds = rioxarray.open_rasterio(tfile).squeeze()
            stack.append(ds.values)

        arr = np.stack(stack, axis=-1)

        for bi, band_name in enumerate(bands):
            layers.append(arr[:, :, bi])
            meta_data.append((date_str, band_name))

    if not layers:
        print("Нет изображений после фильтрации по месяцам.")
        return existing_tensor if flag else None, existing_metadata if flag else None

    new_tensor = normalize_tensor(np.stack(layers, axis=0))

    if flag and existing_tensor is not None:
        if existing_tensor.shape[1:] == new_tensor.shape[1:]:
            tensor = np.concatenate((existing_tensor, new_tensor), axis=0)
            meta_data = (existing_metadata or []) + meta_data
        else:
            print(
                f"Размерности тензоров не совпадают! "
                f"existing_tensor shape: {existing_tensor.shape[1:]}, "
                f"new_tensor shape: {new_tensor.shape[1:]}. "
                f"Новые данные не добавлены."
            )
            tensor = existing_tensor.copy()
            meta_data = existing_metadata.copy() if existing_metadata else []
    else:
        tensor = new_tensor.copy()

    return tensor, meta_data