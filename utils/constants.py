QA_TEMPLATE_WITH_CONTEXT = """Context information is below.
---------------------
{contexts}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:"""

QA_TEMPLATE_WITH_CONTEXT_PREFIX = """Context information is below.
---------------------
{contexts}
---------------------
Given the context information and not prior knowledge, answer the query.
"""

QA_TEMPLATE_WITHOUT_CONTEXT = """Query: {question}
Answer:"""

QA_TEMPLATE_WITHOUT_CONTEXT_PREFIX = """"""

BIOGEN_TEMPLATE_WITH_CONTEXT = """{contexts}

{question}"""

BIOGEN_TEMPLATE_WITH_CONTEXT_PREFIX = """{contexts}

"""

BIOGEN_TEMPLATE_WITHOUT_CONTEXT = """{question}"""

BIOGEN_TEMPLATE_WITHOUT_CONTEXT_PREFIX = """"""
