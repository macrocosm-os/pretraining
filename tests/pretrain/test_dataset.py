import unittest

import numpy as np

from collections import defaultdict
from competitions.data import CompetitionId
from constants import MODEL_CONSTRAINTS_BY_COMPETITION_ID
import pretrain as pt
from neurons import config

# Get the config
config = config.validator_config()


class TestDataset(unittest.TestCase):
    def test_fineweb_loader_page_copy(self):
        """
        Test that pages can be correctly copied from one FineWeb dataloader
        to another
        """
        # Some test params
        NUM_PAGES = 20

        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MODEL],
            cache_dir=config.model_dir,
        )

        # First dataloader
        dataloader_1 = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=4, sequence_length=4092, num_pages=NUM_PAGES, tokenizer=tokenizer
        )

        # Assert that the number of pages loaded successfully are the one required
        self.assertEqual(len(dataloader_1.pages), NUM_PAGES)

        # Now create a second loader without automatic page loading
        dataloader_2 = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=4, sequence_length=4092, num_pages=None, tokenizer=tokenizer
        )

        # Copy pages from the first dataloader
        dataloader_2.fetch_data_for_pages(pages=dataloader_1.pages)

        # Assert both dataloaders have the same pages
        self.assertEqual(set(dataloader_1.pages), set(dataloader_2.pages))

        # Assert that both have the same buffers
        self.assertTrue(np.array_equal(dataloader_1.buffer, dataloader_2.buffer))

    def test_fineweb_loader_unique_pages(self):
        """Tests that the fineweb loader only loads unique pages."""
        # Ensure we get all 5 possible pages of the aritificially shortened data.
        NUM_PAGES = 5
        CONFIG_DATA = {"CC-MAIN-2013-20": {"num_rows": 499, "split": "train"}}

        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MODEL],
            cache_dir=config.model_dir,
        )

        dataloader = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=None, sequence_length=4092, num_pages=None, tokenizer=tokenizer
        )

        dataloader.configs_data = CONFIG_DATA

        # Only fetch these once for performance, although for better correctness should consider running in a loop.
        dataloader._fetch_data_to_buffer(NUM_PAGES)
        self.assertEqual(len(dataloader.pages), NUM_PAGES)
        self.assertEqual(len(set(dataloader.pages)), NUM_PAGES)

        rows = dataloader.fetch_data_to_rows(NUM_PAGES)
        self.assertEqual(len(rows), NUM_PAGES * dataloader.num_rows_per_page)
        self.assertEqual(len(set(rows)), NUM_PAGES * dataloader.num_rows_per_page)

    def test_fineweb_loader_duplicate_threshold(self):
        """Tests that the fineweb loader stops loading after hitting too many duplciate pages."""
        # Try to get 6 pages from a set that only contains 5 pages worth.
        NUM_PAGES = 6
        NUM_PAGES_ACTUAL = 5
        CONFIG_DATA = {"CC-MAIN-2013-20": {"num_rows": 499, "split": "train"}}

        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MODEL],
            cache_dir=config.model_dir,
        )

        dataloader = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=None, sequence_length=4092, num_pages=None, tokenizer=tokenizer
        )

        dataloader.configs_data = CONFIG_DATA

        # Only fetch these once for performance, although for better correctness should consider running in a loop.
        dataloader._fetch_data_to_buffer(NUM_PAGES)
        self.assertEqual(len(dataloader.pages), NUM_PAGES_ACTUAL)
        self.assertEqual(len(set(dataloader.pages)), NUM_PAGES_ACTUAL)

        rows = dataloader.fetch_data_to_rows(NUM_PAGES)
        self.assertEqual(len(rows), NUM_PAGES_ACTUAL * dataloader.num_rows_per_page)
        self.assertEqual(
            len(set(rows)), NUM_PAGES_ACTUAL * dataloader.num_rows_per_page
        )

    def test_fineweb_loader_page_offset(self):
        """Tests that the fineweb loader will only generate page starts that are num rows per pages apart."""
        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MODEL],
            cache_dir=config.model_dir,
        )

        dataloader = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=0, sequence_length=4092, num_pages=0, tokenizer=tokenizer
        )

        # Ensure we know the num_rows_per_page.
        test_num_rows_per_page = 100
        dataloader.num_rows_per_page = test_num_rows_per_page

        # Create a fake configs data with only 599 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 599, "split": "train"}
        }

        # Ensure get random pages returns only 0, 100, 200, 300, 400 and 500
        expected_page_starts_1 = {0, 100, 200, 300, 400, 500}
        page_starts_1 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1)
            _, page_start, _ = random_pages[0]
            page_starts_1[page_start] += 1

        self.assertEqual(set(page_starts_1.keys()), expected_page_starts_1)

        # Create a fake configs data with only 598 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 598, "split": "train"}
        }

        # Ensure get random pages returns only 0, 100, 200, 300, and 400 (since 500-598 is not 100 rows)
        expected_page_starts_2 = {0, 100, 200, 300, 400}
        page_starts_2 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1)
            _, page_start, _ = random_pages[0]
            page_starts_2[page_start] += 1

        self.assertEqual(set(page_starts_2.keys()), expected_page_starts_2)
