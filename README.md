# Live Earth globe
Draggable 3D globe textured with a near-real-time composite of geostationary GeoColor imagery
(GOES-19, GOES-18 via NOAA STAR; Himawari-9, GK-2A, Meteosat 0°, Meteosat-9 IODC via CIRA/RAMMB SLIDER),
reprojected to equirectangular and blended, Blue Marble fill at the poles.
A scheduled GitHub Action runs `build.py` every ~15 min and deploys `site/` to GitHub Pages.
Local: `pip install numpy pillow && python build.py && python -m http.server -d site`.
