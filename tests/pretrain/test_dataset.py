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
    def test_fineweb_loader_unique_pages(self):
        """Tests that the fineweb loader only loads unique pages."""
        # Ensure we get all the possible pages of the aritificially shortened data.
        NUM_PAGES = 5
        NUM_ROWS_PER_PAGE = 100
        NUM_SAMPLES = 1000
        CONFIG_DATA = {"CC-MAIN-2013-20": {"num_rows": 499, "split": "train"}}

        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
            cache_dir=config.model_dir,
        )

        dataloader = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=None, sequence_length=4096, num_samples=None, tokenizer=tokenizer
        )

        dataloader.configs_data = CONFIG_DATA
        dataloader.num_rows_per_page = NUM_ROWS_PER_PAGE

        # Only fetch these once for performance, although for better correctness should consider running in a loop.
        dataloader._fetch_data_to_buffer(NUM_SAMPLES)
        # Check that we did not reach the target num samples, but that we still stopped and did not load duplicates.
        self.assertLess(dataloader._get_loaded_sample_count(), NUM_SAMPLES)
        # We check for max pages or max pages - 1 to handle the random offset.
        self.assertIn(len(dataloader.pages), [NUM_PAGES, NUM_PAGES - 1])
        self.assertIn(len(set(dataloader.pages)), [NUM_PAGES, NUM_PAGES - 1])

    def test_fineweb_loader_page_offset(self):
        """Tests that the fineweb loader will only generate page starts that are num rows per pages apart."""
        # Load a tokenizer.
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
            cache_dir=config.model_dir,
        )

        dataloader = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=0, sequence_length=4096, num_samples=0, tokenizer=tokenizer
        )

        # Ensure we know the num_rows_per_page.
        test_num_rows_per_page = 100
        dataloader.num_rows_per_page = test_num_rows_per_page

        # Create a fake configs data with only 599 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 599, "split": "train"}
        }

        # Ensure get random pages returns only 0, 100, 200, 300, 400 and 500.
        expected_page_starts_1 = {0, 100, 200, 300, 400, 500}
        page_starts_1 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=0)
            _, page_start, _ = random_pages[0]
            page_starts_1[page_start] += 1

        self.assertEqual(set(page_starts_1.keys()), expected_page_starts_1)

        # Create a fake configs data with only 598 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 598, "split": "train"}
        }

        # Ensure get random pages returns only 0, 100, 200, 300, and 400 (since 500-598 is not 100 rows).
        expected_page_starts_2 = {0, 100, 200, 300, 400}
        page_starts_2 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=0)
            _, page_start, _ = random_pages[0]
            page_starts_2[page_start] += 1

        self.assertEqual(set(page_starts_2.keys()), expected_page_starts_2)

    def test_fineweb_loader_page_initial_offset(self):
        """Tests that the fineweb loader correctly applies an initial offset to the page starts."""
        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
            cache_dir=config.model_dir,
        )

        dataloader = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=0, sequence_length=4096, num_samples=0, tokenizer=tokenizer
        )

        # Ensure we know the num_rows_per_page.
        test_num_rows_per_page = 100
        dataloader.num_rows_per_page = test_num_rows_per_page

        # Create a fake configs data with only 599 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 599, "split": "train"}
        }

        # Define initial offset of 50.
        initial_offset = 50
        # Ensure get random pages returns only 50, 150, 250, 350, and 450.
        expected_page_starts_1 = {50, 150, 250, 350, 450}
        page_starts_1 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=initial_offset)
            _, page_start, _ = random_pages[0]
            page_starts_1[page_start] += 1

        self.assertEqual(set(page_starts_1.keys()), expected_page_starts_1)

        # Create a fake configs data with only 548 rows.
        dataloader.configs_data = {
            "CC-MAIN-2013-20": {"num_rows": 548, "split": "train"}
        }

        # Ensure get random pages returns only 50, 150, 250, and 350 (since 450-548 is not 100 rows)
        expected_page_starts_2 = {50, 150, 250, 350}
        page_starts_2 = defaultdict(int)
        for _ in range(1000):
            random_pages = dataloader.get_random_pages(1, initial_offset=initial_offset)
            _, page_start, _ = random_pages[0]
            page_starts_2[page_start] += 1

        self.assertEqual(set(page_starts_2.keys()), expected_page_starts_2)

    def test_fineweb_loader_seed(self):
        """Tests that the fineweb data loader fetches the same data with the same seed (barring retries)."""

        # Use the same seed for each loader.
        RANDOM_SEED = 1
        # Fetch just 25 samples to keep the test reasonably fast.
        NUM_SAMPLES = 25

        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
            cache_dir=config.model_dir,
        )

        # First dataloader.
        dataloader_1 = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=4,
            sequence_length=4096,
            num_samples=NUM_SAMPLES,
            tokenizer=tokenizer,
            random_seed=RANDOM_SEED,
        )

        # Assert that the number of samples requested were loaded.
        self.assertGreaterEqual(dataloader_1._get_loaded_sample_count(), NUM_SAMPLES)

        # Now create a second loader with the same seed.
        dataloader_2 = pt.dataset.SubsetFineWebEdu2Loader(
            batch_size=4,
            sequence_length=4096,
            num_samples=NUM_SAMPLES,
            tokenizer=tokenizer,
            random_seed=RANDOM_SEED,
        )

        # Assert both dataloaders have the same pages and sample counts
        self.assertEqual(set(dataloader_1.pages), set(dataloader_2.pages))
        self.assertEqual(
            dataloader_1._get_loaded_sample_count(),
            dataloader_2._get_loaded_sample_count(),
        )

        # Assert that both have the same buffers
        self.assertTrue(np.array_equal(dataloader_1.buffer, dataloader_2.buffer))

    def test_falcon_loader_seed(self):
        """Tests that the falcon data loader fetches the same data with the same seed."""

        # Use the same seed for each loader.
        RANDOM_SEED = 1
        # Fetch just 25 samples to keep the test reasonably fast.
        NUM_SAMPLES = 25

        # Load a tokenizer
        tokenizer = pt.model.load_tokenizer(
            MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
            cache_dir=config.model_dir,
        )

        # First dataloader.
        dataloader_1 = pt.dataset.SubsetFalconLoader(
            batch_size=4,
            sequence_length=4096,
            num_samples=NUM_SAMPLES,
            tokenizer=tokenizer,
            random_seed=RANDOM_SEED,
        )

        # Assert that the number of samples requested were loaded.
        self.assertGreaterEqual(dataloader_1._get_loaded_sample_count(), NUM_SAMPLES)

        # Now create a second loader with the same seed.
        dataloader_2 = pt.dataset.SubsetFalconLoader(
            batch_size=4,
            sequence_length=4096,
            num_samples=NUM_SAMPLES,
            tokenizer=tokenizer,
            random_seed=RANDOM_SEED,
        )

        # Assert both dataloaders have the same pages and sample counts
        self.assertEqual(set(dataloader_1.pages), set(dataloader_2.pages))
        self.assertEqual(
            dataloader_1._get_loaded_sample_count(),
            dataloader_2._get_loaded_sample_count(),
        )
        # Assert that both have the same buffers
        self.assertTrue(np.array_equal(dataloader_1.buffer, dataloader_2.buffer))
