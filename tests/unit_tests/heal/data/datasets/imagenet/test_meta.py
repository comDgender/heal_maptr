from heal.data.datasets.imagenet.meta import Classes


def test_imagenet_classes_basic_shape():
    assert isinstance(Classes, dict)
    assert len(Classes) == 1000
    assert min(Classes.keys()) == 0
    assert max(Classes.keys()) == 999


def test_imagenet_classes_values_are_strings():
    assert isinstance(Classes[0], str)
    assert isinstance(Classes[999], str)
