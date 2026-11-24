def test_datasets_package_importable():
    import heal.data.datasets as datasets_pkg

    assert datasets_pkg is not None


def test_imagenet_package_importable():
    import heal.data.datasets.imagenet as imagenet_pkg

    assert imagenet_pkg is not None
