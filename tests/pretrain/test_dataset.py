import unittest

import numpy as np

import pretrain as pt
from neurons import config

# Get the config
config = config.validator_config()

def test_FineWeb_loader_page_copy():
    """
    Test that pages can be correctly copied from one FineWeb dataloader 
    to another
    """
    # Some test params
    NUM_PAGES = 20
    
    # Load a tokenizer
    tokenizer = pt.model.get_tokenizer(cache_dir=config.model_dir)    

    # First dataloader
    dataloader_1 = pt.dataset.SubsetFineWebEdu2Loader(
        batch_size=4,
        sequence_length=4092,
        num_pages=NUM_PAGES,
        tokenizer=tokenizer)

    # Assert that the number of pages loaded successfully are the one required
    assert len(dataloader_1.pages) == NUM_PAGES


    # Now create a second loader without automatic page loading
    dataloader_2 = pt.dataset.SubsetFineWebEdu2Loader(
        batch_size=4,
        sequence_length=4092,
        num_pages=None,
        tokenizer=tokenizer)
    
    # Copy pages from the first dataloader
    dataloader_2.fetch_data_for_pages(pages=dataloader_1.pages)

    # Assert both dataloaders have the same pages
    assert set(dataloader_1.pages) == set(dataloader_2.pages)

    # Assert that both have the same buffers
    assert np.array_equal(dataloader_1.buffer, dataloader_2.buffer)
