from pathlib import Path

from qtpy.QtWidgets import QApplication

from cali.gui._init_dialog import _InputDialog
from cali.readers import TensorstoreZarrReader
from cali.readers._tiff_collection_reader import TiffCollectionReader

app = QApplication([])
init_dialog = _InputDialog(data_path="/Users/fdrgsp/Desktop/cali_test/tiffs")
# init_dialog = _InputDialog(data_path="/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr")

init_dialog.resize(700, init_dialog.sizeHint().height())
if init_dialog.exec():
    value = init_dialog.value()
    # input from database
    if value.data_path is not None:
        data = None
        if (d_path := Path(value.data_path)).is_dir():
            try:
                data = TensorstoreZarrReader(d_path)
                print("✅ Successfully initialized from database")
            except Exception:
                try:
                    custom_files = sorted(
                        list(d_path.glob("*.tif")) + list(d_path.glob("*.tiff"))
                    )
                    file_map = {}
                    for i in sorted(custom_files):
                        well, fov = i.stem.split("_")
                        if well not in file_map:
                            file_map[well] = []
                        if i not in file_map[well]:
                            file_map[well].append(i)
                        data = TiffCollectionReader(
                            file_map=file_map,
                            plate="96-well",
                            metadata={
                                "exposure_ms": 100.0,
                                "pixel_size_um": 0.65,
                            },
                        )
                except Exception as e:
                    print(f"❌ Failed to initialize from database:\n{e}")
        print(data)
        app.quit()

app.exec()

