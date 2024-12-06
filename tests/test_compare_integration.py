import pytest
import os
from logbatcher.compare_clustering import process_all_datasets
from pathlib import Path

@pytest.fixture
def mock_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "mock_dataset"
    dataset_dir.mkdir()
    log1 = dataset_dir / "sample1.log"
    log2 = dataset_dir / "sample2.log"
    log3 = dataset_dir / "sample3.log"
    log1.write_text("2024-01-01 INFO This is a test log message 10.0.0.1\n")
    log2.write_text("ERROR user=1234 count=9999 http://example.com Another message\n")
    log3.write_text("2024-02-02 DEBUG Another test log 192.168.1.1\n")

    return tmp_path

def test_process_all_datasets(mock_dataset_dir):
    process_all_datasets(datasets_dir=str(mock_dataset_dir))
    recommendations_file = Path("cluster_recommendation.txt")
    assert recommendations_file.exists()

    compare_plots_dir = Path("compare_plots")
    assert compare_plots_dir.is_dir()
    png_files = list(compare_plots_dir.glob("*.png"))
    assert len(png_files) > 0

    content = recommendations_file.read_text()
    assert "Ranked Methods per Dataset:" in content
    assert "Dataset:" in content

    # Cleanup after test
    recommendations_file.unlink()
    for file in compare_plots_dir.iterdir():
        file.unlink()
    compare_plots_dir.rmdir()
