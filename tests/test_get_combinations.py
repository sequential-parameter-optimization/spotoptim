# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for spotoptim.utils.stats.get_combinations
"""

import pytest

from spotoptim.utils.stats import get_combinations


class TestGetCombinationsBasic:
    def test_indices_basic(self):
        ind_list = [0, 10, 20, 30]
        combos = get_combinations(ind_list, type="indices")
        assert combos == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def test_values_basic(self):
        ind_list = [0, 10, 20, 30]
        combos = get_combinations(ind_list, type="values")
        assert combos == [(0, 10), (0, 20), (0, 30), (10, 20), (10, 30), (20, 30)]

    def test_empty_list(self):
        combos = get_combinations([], type="indices")
        assert combos == []

    def test_single_element(self):
        combos = get_combinations([42], type="indices")
        assert combos == []


class TestGetCombinationsEdgeCases:
    def test_two_elements_indices(self):
        combos = get_combinations([1, 2], type="indices")
        assert combos == [(0, 1)]

    def test_two_elements_values(self):
        combos = get_combinations([1, 2], type="values")
        assert combos == [(1, 2)]

    def test_non_list_input_raises(self):
        with pytest.raises(ValueError):
            get_combinations((1, 2, 3), type="indices")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_combinations([0, 1, 2], type="wrong")


class TestGetCombinationsOrdering:
    def test_order_is_combinations_without_repeats_indices(self):
        combos = get_combinations([0, 1, 2, 3], type="indices")
        # Check length and no repeated or reversed pairs
        assert len(combos) == 6
        assert (0, 1) in combos and (1, 0) not in combos
        assert (2, 3) in combos and (3, 2) not in combos

    def test_order_is_combinations_without_repeats_values(self):
        lst = [0, 1, 2, 3]
        combos = get_combinations(lst, type="values")
        assert len(combos) == 6
        assert (0, 1) in combos and (1, 0) not in combos
        assert (2, 3) in combos and (3, 2) not in combos


class TestDocstringExamples:
    def test_doc_example_indices(self):
        ind_list = [0, 10, 20, 30]
        combos = get_combinations(ind_list)
        assert combos == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def test_doc_example_values(self):
        ind_list = [0, 10, 20, 30]
        combos = get_combinations(ind_list, type="values")
        assert combos == [(0, 10), (0, 20), (0, 30), (10, 20), (10, 30), (20, 30)]
