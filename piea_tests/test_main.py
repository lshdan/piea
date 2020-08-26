
import pytest
import pathlib

@pytest.fixture()
def data_dir():
    return pathlib.Path(__file__).parent / 'data'

def test_main(data_dir):
    from piea.app import main, parse_args

    args = parse_args([
        
        # '--guider=resnet',
        '--guider=mobilenet',
        '--tgt={}'.format((data_dir / "test_1_out").as_posix()),
        '--index=index.txt',
        '--loss=2', 
        '--src={}'.format((data_dir / "test_1").as_posix())])

    assert args is not None

    main(args)
