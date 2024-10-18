from pydantic import BaseModel
from typing import List


class Assumption(BaseModel):
    """
    An assumption is a condition that must be met in order for
    a statistical test to be valid.
    Contains the one sentence description of the assumption, and a boolean pass_status.
    """

    description: str
    pass_status: bool


class TestExample(BaseModel):
    """
    A test example is an example of how to apply a statistical test to a situation.
    Contains the situation, the test name, the description of the test, the assumptions,
    and the steps to apply the test.

    Also contains notes for any details the user should be aware of,
    such as potential pitfalls or
    nuances that are important to understand.
    """

    situation: str
    test_name: str
    description: str
    assumption_descriptions: List[str]
    assumption_pass_statuses: List[bool]
    check_assumptions: str
    apply_test: str
    notes: str


class TestRecommendation(BaseModel):
    """
    A test recommendation is a recommendation for a statistical test to apply
    to a situation.
    Contains the test name, the assumptions, and the steps to apply the test.
    """

    test_name: str
    assumptions_descriptions: List[str]
    assumptions_pass_statuses: List[bool]


class FindTestResponse(BaseModel):
    """
    A find test response is a response to a query about finding an appropriate
    statistical test to apply to a situation.
    Contains the situation, and a list of recommended tests.
    """

    situation: str
    recommended_tests: List[TestRecommendation]
