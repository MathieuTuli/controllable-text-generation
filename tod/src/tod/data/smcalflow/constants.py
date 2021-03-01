"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
--
Semantic Machines\N{TRADE MARK SIGN} software.

Creates text data (source-target pairs) to be used for training OpenNMT models.
"""


class SpecialStrings:
    """Special strings in stringified turn parts.
    """

    # empty value (we need it since some library doesn't like an empty string)
    NULL = "__NULL"
    # indicates there is a break between the two utterance segments
    BREAK = "__BREAK"
    # indicates the user is the speaker for the following utterance
    SPEAKER_USER = "__User"
    # indicates the agent is the speaker for the following utterance
    SPEAKER_AGENT = "__Agent"
    # start of a program
    START_OF_PROGRAM = "__StartOfProgram"
