import unittest

from concept_extractor import extract_concepts
from question_generator import _quality_checked_topic


class ConceptFilterRegression(unittest.TestCase):

    BLOCKED_TOPICS = [
        "MHz clock 8085 upward compatible",
        "as input or as output ports",
        "Briefly explain",
        "Discuss the key differences between",
        "Decoder decodes instruction in IR",
        "iscomputed as 10 DS SI",
        "25 MHz externally",
        "Engineering as",
        "Robert Jac kall",
        "Experimentation Engineers as responsible",
        "above you wants from you",
    ]

    VALID_TOPICS = [
        "The effective address",
        "8085 microprocessor",
        "8255 programmable peripheral interface",
    ]

    def test_quality_checked_topic_rejects_blockers(self):
        for topic in self.BLOCKED_TOPICS:
            self.assertEqual(_quality_checked_topic(topic), "", topic)

    def test_quality_checked_topic_accepts_clean_topics(self):
        cleaned = [_quality_checked_topic(topic) for topic in self.VALID_TOPICS]
        self.assertIn("effective address", cleaned)
        self.assertIn("8085 microprocessor", cleaned)
        self.assertIn("8255 programmable peripheral interface", cleaned)

    def test_extract_concepts_filtering(self):
        raw = "\n".join(
            [
                "10. Briefly explain MHz clock 8085 upward compatible.",
                "3. Explain the applications of as input or as output ports with suitable examples.",
                "4. Discuss The effective address with suitable examples.",
                "5. Briefly explain Decoder decodes instruction in IR.",
                "6. Briefly explain iscomputed as 10 DS SI.",
                "7. State the applications of Engineering as.",
                "8. Define Robert Jac kall.",
                "9. Briefly explain Experimentation Engineers as responsible.",
                "10. Briefly explain above you wants from you.",
            ]
        )
        concepts = extract_concepts(raw)
        self.assertIn("effective address", concepts)
        self.assertNotIn("Briefly explain", concepts)
        self.assertNotIn("MHz clock 8085 upward compatible", concepts)
        self.assertNotIn("as input or as output ports", concepts)
        self.assertNotIn("Engineering as", concepts)
        self.assertNotIn("Robert Jac kall", concepts)
        self.assertNotIn("Experimentation Engineers as responsible", concepts)
        self.assertNotIn("above you wants from you", concepts)


if __name__ == "__main__":
    unittest.main()
