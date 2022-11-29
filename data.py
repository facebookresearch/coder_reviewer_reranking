# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import json
import os
import regex


class NL2BashDataset(object):
    def __init__(self, path="dataset/nl2bash/data/bash"):
        self.data = collections.defaultdict()
        for split in ["train", "dev", "test"]:
            nls = [x.strip() for x in open(os.path.join(path, f"{split}.nl.filtered"))]
            cms = [x.strip() for x in open(os.path.join(path, f"{split}.cm.filtered"))]
            infos = ["" for x in open(os.path.join(path, f"{split}.cm.filtered"))]
            self.data[split] = list(zip(nls, cms, infos))


class SpiderDataset(object):
    def __init__(self, path="dataset/spider"):
        self.data = collections.defaultdict()
        self.dbs = json.load(open(f"{path}/tables.json"))
        self.id2db = {item["db_id"]: item for item in self.dbs}
        for split in ["train", "dev"]:
            split_fname = "train_spider" if split == "train" else split
            data = json.load(open(f"{path}/{split_fname}.json"))
            nls = [x["question"] for x in data]
            cms = [x["query"] for x in data]
            db_info = [self.extract_db_info(x["db_id"]) for x in data]
            self.data[split] = list(zip(nls, cms, db_info))

    def extract_db_info(self, db_id):
        db = self.id2db[db_id]
        id2table = {
            i: table_name for i, table_name in enumerate(db["table_names_original"])
        }
        info = f"{db_id} "
        used_table_id = set()
        for table_id, column_name in db["column_names_original"]:
            if table_id == -1:
                info += f"| {column_name} "
            elif table_id not in used_table_id:
                info += f"| {id2table[table_id]} : {column_name} "
                used_table_id.add(table_id)
            else:
                info += f", {column_name} "
        return info.strip()


class MBPPGoogleDataset(object):
    def __init__(self, path="dataset/mbpp/mbpp.jsonl", mode="function_name"):
        raw_data = sorted(
            [json.loads(x) for x in open(path)], key=lambda x: x["task_id"]
        )
        for i, data_item in enumerate(raw_data):
            assert data_item["task_id"] == i + 1
        self.raw_data = collections.defaultdict()
        self.mode = mode
        # 374 for training, 100 heldout, 500 test
        self.raw_data["train"] = raw_data[:10] + raw_data[510:]
        self.raw_data["test"] = raw_data[10:510]
        # data for codex collector, in input-output-info format
        self.data = collections.defaultdict()
        for split in self.raw_data:
            self.data[split] = self.extract_data(self.raw_data[split], mode)

    @staticmethod
    def extract_data(raw_data, mode):
        if mode == "function_name":
            get_function_name = lambda test_example: regex.match(
                "assert [\(]*([^\(]+)\(", test_example
            ).group(1)
            info = [get_function_name(x["test_list"][0]) for x in raw_data]
        elif mode == "assertion":
            info = [x["test_list"][0] for x in raw_data]
        elif mode == "assertion-full":
            info = [x["test_list"] for x in raw_data]
        else:
            raise Exception(f"Mode {mode} not supported.")
        nls = [x["text"] for x in raw_data]
        codes = [x["code"] for x in raw_data]
        return list(zip(nls, codes, info))


from dataset.human_eval.human_eval.data import read_problems


class HumanEvalDataset(object):
    def __init__(
        self,
        path="dataset/human_eval/dataset/HumanEval.jsonl",
        assertion_path="",
        mode="assertion",
    ):
        self.path = path
        self.data = dict()
        self.raw_data = read_problems(path)
        self.mode = mode
        if assertion_path != "":
            self.assertion_data = read_problems(assertion_path)
        else:
            self.assertion_data = self.raw_data

        self.data["test"] = self.extract_data()

    def extract_data(self):
        nls = []
        codes = []
        info = []
        for pid, prob in self.raw_data.items():
            assert_prob = self.assertion_data[pid]
            nls.append(prob["prompt"])
            docstring, func_header, func_context, doc_start = extract_docstring(
                assert_prob["prompt"]
            )
            self.raw_data[pid]["func_header"] = func_header.strip() + "\n"
            self.raw_data[pid]["func_context"] = func_context
            codes.append(prob["canonical_solution"])

            if self.mode != "prompt_only":
                assertions = extract_test(pid, prob["entry_point"], docstring)
                if self.mode == "assertion":
                    self.raw_data[pid]["assertion"] = assertions[0]
                    info.append(assertions[0])
                else:
                    self.raw_data[pid]["assertion"] = assertions
                    info.append(assertions)
            else:
                info.append([])
        return list(zip(nls, codes, info))


class MBPPSanDataset(HumanEvalDataset):
    def extract_data(self):
        nls = []
        codes = []
        info = []
        for pid, prob in self.raw_data.items():
            nls.append(prob["prompt"])
            docstring, func_header, func_context, doc_start = extract_docstring(
                prob["prompt"]
            )
            self.raw_data[pid]["func_header"] = func_header.strip() + "\n"
            self.raw_data[pid]["func_context"] = func_context
            codes.append(prob["canonical_solution"])

            if self.mode != "prompt_only":
                assertions = [
                    l.strip() for l in prob["test"].split("\n")[1:] if l.strip() != ""
                ]
                if self.mode == "assertion":
                    self.raw_data[pid]["assertion"] = assertions[0]
                    info.append(assertions[0])
                elif self.mode == "assertion-all":
                    self.raw_data[pid]["assertion"] = assertions
                    info.append(assertions)
                else:
                    raise ValueError("invalid mode")
            else:
                info.append([])
        return list(zip(nls, codes, info))


def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1


def _check_test_case_validation(test_case):
    if len(test_case.strip()) < 1:
        return False
    if "assert" not in test_case:
        return False
    try:
        multi_line_test_case = test_case.replace("\n", "\n    ")
        assert_in_a_block = f"try:\n    {multi_line_test_case}\nexcept:\n    pass\n"
        compile(assert_in_a_block, "", "exec")
        return True
    except Exception:
        return False


def extract_generated_tests(content, entry_point):
    def _truncate(content):
        for identifier in ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]:
            if identifier in content:
                content = content.split(identifier)[0]
        return content.strip()

    split_by_assert = [
        f"assert {part}".strip()
        for part in f"assert {content}".split("assert ")
        if (entry_point.strip() in part) and len(part.strip()) > 0
    ]
    truncated_test_cases = [_truncate(i) for i in split_by_assert]
    checked_assertions = [
        i for i in truncated_test_cases if _check_test_case_validation(i)
    ]
    return checked_assertions


def extract_docstring(prompt):
    func_start = max(rindex(prompt, " fed") - 4, 0)
    clean_prompt = prompt[func_start:]
    if '"""' in prompt:
        doc_start = '"""'
    else:
        doc_start = "'''"
    docstring = clean_prompt[clean_prompt.strip().index(doc_start) :]
    func_header = clean_prompt[: clean_prompt.strip().index(doc_start)]
    func_context = prompt[:func_start]
    return docstring, func_header, func_context, doc_start


def extract_test(pid, func_name, docstring):
    if pid in manual_extract:
        return manual_extract[pid]
    else:
        return _extract_tests(func_name, docstring)


def _extract_tests(func_name, docstring):
    all_tests = []
    doc_lines = docstring.strip().split("\n")

    test_start = False

    if ">>>" in docstring:
        for l in doc_lines:
            if not test_start:
                if ">>>" in l and func_name in l:
                    test_start = True
            if test_start:
                if ">>>" in l and func_name in l:
                    l = l.strip()[3:].strip()
                    all_tests.append(l)
                elif l.strip() != "" and '"""' not in l:
                    all_tests[-1] = "assert " + all_tests[-1] + f" == {l.strip()}"
                    test_start = False
    elif any(
        ["==>" in docstring, "=>" in docstring, "->" in docstring, "➞" in docstring]
    ):
        for special_char in ["==>", "=>", "->", "➞", "==>"]:
            if special_char in docstring:
                break
        for l in doc_lines:
            if not test_start:
                if special_char in l and func_name in l:
                    test_start = True
            if test_start and (special_char in l and func_name in l):
                l = l.strip().replace(special_char, "==")
                l = "assert " + l
                all_tests.append(l)
    elif any(["==" in docstring, "returns" in docstring]):
        for special_char in ["==", "returns"]:
            if special_char in docstring:
                break
        for l in doc_lines:
            if not test_start:
                if special_char in l and func_name + "(" in l:
                    test_start = True
            if test_start and (special_char in l and func_name in l):
                l = "assert " + l.strip().replace(special_char, "==")
                all_tests.append(l)

    return all_tests


manual_extract = {
    "HumanEval/12": [
        "assert longest(['a', 'b', 'c']) == 'a'",
        "assert longest(['a', 'bb', 'ccc']) == 'ccc'",
    ],
    "HumanEval/38": ["assert True == True"],  # empty assertion to handle no doc test
    "HumanEval/41": ["assert True == True"],  # empty assertion to handle no doc test
    "HumanEval/50": ["assert True == True"],  # empty assertion to handle no doc test
    "HumanEval/67": [
        'assert fruit_distribution("5 apples and 6 oranges", 19) == 8'
        'assert fruit_distribution("0 apples and 1 oranges",3) == 2'
        'assert fruit_distribution("2 apples and 3 oranges", 100) == 95'
        'assert fruit_distribution("100 apples and 1 oranges",120) == 19'
    ],
    "HumanEval/68": [
        "assert pluck([4,2,3]) == [2, 1]",
        "assert pluck([1,2,3]) == [2, 1]",
        "assert pluck([]) == []",
        "assert pluck([5, 0, 3, 0, 4, 2]) == [0, 1]",
    ],
    "HumanEval/78": [
        "assert hex_key('AB') == 1",
        "assert hex_key('1077E') == 2",
        "assert hex_key('ABED1A33') == 4",
        "assert hex_key('123456789ABCDEF0') == 6",
        "assert hex_key('2020') == 2",
    ],
    "HumanEval/79": [
        "assert decimal_to_binary(15) == 'db1111db'",
        "assert decimal_to_binary(32) == 'db100000db'",
    ],
    "HumanEval/81": [
        "assert grade_equation([4.0, 3, 1.7, 2, 3.5]) ==> ['A+', 'B', 'C-', 'C', 'A-']"
    ],
    "HumanEval/83": ["assert True == True"],  # empty assertion to handle no doc test
    "HumanEval/84": ["assert True == True"],  # empty assertion to handle no doc test
    "HumanEval/86": [
        "assert anti_shuffle('Hi') == 'Hi'",
        "assert anti_shuffle('hello') == 'ehllo'",
        "assert anti_shuffle('Hello World!!!') == 'Hello !!!Wdlor'",
    ],
    "HumanEval/88": [
        "assert sort_array([]) == []",
        "assert sort_array([5]) == [5]",
        "assert sort_array([2, 4, 3, 0, 1, 5]) == [0, 1, 2, 3, 4, 5]",
        "assert sort_array([2, 4, 3, 0, 1, 5, 6]) == [6, 5, 4, 3, 2, 1, 0]",
    ],
    "HumanEval/94": [
        "assert skjkasdkd([0,3,2,1,3,5,7,4,5,5,5,2,181,32,4,32,3,2,32,324,4,3]) == 10",
        "assert skjkasdkd([1,0,1,8,2,4597,2,1,3,40,1,2,1,2,4,2,5,1]) == 25",
        "assert skjkasdkd([1,3,1,32,5107,34,83278,109,163,23,2323,32,30,1,9,3]) == 13",
        "assert skjkasdkd([0,724,32,71,99,32,6,0,5,91,83,0,5,6]) == 11",
        "assert skjkasdkd([0,81,12,3,1,21]) == 3",
        "assert skjkasdkd([0,8,1,2,1,7]) == 7",
    ],
    "HumanEval/95": [
        'assert check_dict_case({"a":"apple", "b":"banana"}) == True.',
        'assert check_dict_case({"a":"apple", "A":"banana", "B":"banana"}) == False.',
        'assert check_dict_case({"a":"apple", 8:"banana", "a":"apple"}) == False.',
        'assert check_dict_case({"Name":"John", "Age":"36", "City":"Houston"}) == False.',
        'assert check_dict_case({"STATE":"NC", "ZIP":"12345" }) == True.',
    ],
    "HumanEval/97": [
        "assert multiply(148, 412) == 16",
        "assert multiply(19, 28) == 72",
        "assert multiply(2020, 1851) == 0",
        "assert multiply(14,-15) == 20",
    ],
    "HumanEval/102": [
        "assert choose_num(12, 15) == 14",
        "assert choose_num(13, 12) == -1",
    ],
    "HumanEval/105": ["assert True == True"],
    "HumanEval/107": [
        "assert even_odd_palindrome(3) == (1, 3)",
        "assert even_odd_palindrome(12) == (4, 6)",
    ],
    "HumanEval/108": [
        "assert count_nums([]) == 0",
        "assert count_nums([-1, 11, -11]) == 1",
        "assert count_nums([1, 1, 2]) == 3",
    ],
    "HumanEval/115": [
        "assert max_fill([[0,0,1,0], [0,1,0,0], [1,1,1,1]]) == 1",
        "assert max_fill([[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]) == 2",
        "assert max_fill([[0,0,0], [0,0,0]]) == 0",
    ],
    "HumanEval/116": [
        "assert sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]",
        "assert sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]",
        "assert sort_array([1, 0, 2, 3, 4]) == [0, 1, 2, 3, 4]",
    ],
    "HumanEval/112": [
        "assert reverse_delete('abcde', 'ae') == ('bcd',False)",
        "assert reverse_delete('abcdef', 'b') == ('acdef',False)",
        "assert reverse_delete('abcdedcba', 'ab') == ('cdedc',True)",
    ],
    "HumanEval/120": [
        "assert maximum([-3, -4, 5], 3) == [-4, -3, 5]",
        "assert maximum([4, -4, 4], 2) == [4, 4]",
        "assert maximum([-3, 2, 1, 2, -1, -2, 1], 1) == [2]",
    ],
    "HumanEval/122": [
        "assert add_elements([111,21,3,4000,5,6,7,8,9]) == 24",
    ],
    "HumanEval/128": [
        "assert prod_signs([1, 2, 2, -4]) == -9",
        "assert prod_signs([0, 1]) == 0",
        "assert prod_signs([]) == None",
    ],
    "HumanEval/129": [
        "assert minPath([[1,2,3], [4,5,6], [7,8,9]], 3) == [1, 2, 1]",
        "assert minPath([[5,9,3], [4,1,6], [7,8,2]], 1) == [1]",
    ],
    "HumanEval/130": ["assert tri(3) == [1, 3, 2, 8]"],
    "HumanEval/133": [
        "assert sum_squares([1,2,3]) == 14",
        "assert sum_squares([1,4,9]) == 98",
        "assert sum_squares([1,3,5,7]) == 84",
        "assert sum_squares([1.4,4.2,0]) == 29",
        "assert sum_squares([-2.4,1,1]) == 6",
    ],
    "HumanEval/135": [
        "assert can_arrange([1,2,4,3,5]) == 3",
        "assert can_arrange([1,2,3]) == -1",
    ],
    "HumanEval/141": [
        "assert file_name_check('example.txt') == 'Yes'",
        "assert file_name_check('1example.dll') == 'No'",
    ],
    "HumanEval/142": [
        "assert sum_squares([1,2,3]) == 6",
        "assert sum_squares([]) == 0",
        "assert sum_squares([-1,-5,2,-1,-5]) == -126",
    ],
    "HumanEval/143": [
        "assert words_in_sentence('This is a test') == 'is'",
        "assert words_in_sentence('lets go for swimming') == 'go for'",
    ],
    "HumanEval/144": [
        'assert simplify("1/5", "5/1") == True',
        'assert simplify("1/6", "2/1") == False',
        'assert simplify("7/10", "10/2") == False',
    ],
    "HumanEval/145": [
        "assert order_by_points([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11]",
        "assert order_by_points([]) == []",
    ],
    "HumanEval/156": [
        "assert int_to_mini_roman(19) == 'xix'",
        "assert int_to_mini_roman(152) == 'clii'",
        "assert int_to_mini_roman(426) == 'cdxxvi'",
    ],
    "HumanEval/147": [
        "assert get_max_triples(5) == 1",
    ],
    "HumanEval/149": [
        'assert list_sort(["aa", "a", "aaa"]) == ["aa"]',
        'assert list_sort(["ab", "a", "aaa", "cd"]) == ["ab", "cd"]',
    ],
    "HumanEval/159": [
        "assert eat(5, 6, 10) == [11, 4]",
        "assert eat(4, 8, 9) == [12, 1]",
        "assert eat(1, 10, 10) == [11, 0]",
        "assert eat(2, 11, 5) == [7, 0]",
    ],
    "HumanEval/160": [
        "assert do_algebra([2, 3, 4, 5], ['+', '*', '-']) == 9",
    ],
    "HumanEval/161": [
        'assert solve("1234") == "4321"',
        'assert solve("ab") == "AB"',
        'assert solve("#a@C") == "#A@c"',
    ],
    "HumanEval/162": [
        "assert string_to_md5('Hello world') == '3e25960a79dbc69b674cd4ec67a72c62'"
    ],
}
