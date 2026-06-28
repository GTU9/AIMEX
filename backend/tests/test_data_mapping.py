import unittest

from app.utils.data_mapping import DataMapper, create_character_data


class DataMapperRegressionTests(unittest.TestCase):
    def test_gender_mappings_preserve_known_values_and_defaults(self):
        self.assertEqual(DataMapper.map_gender_to_vllm("남성"), "MALE")
        self.assertEqual(DataMapper.map_gender_to_vllm("FEMALE"), "FEMALE")
        self.assertEqual(DataMapper.map_gender_to_vllm(None), "NON_BINARY")
        self.assertEqual(DataMapper.map_gender_to_db("여"), 2)
        self.assertEqual(DataMapper.map_gender_to_db("unknown"), 2)

    def test_age_group_boundaries_and_invalid_default(self):
        self.assertEqual(DataMapper.map_age_to_group("19"), 1)
        self.assertEqual(DataMapper.map_age_to_group("20"), 2)
        self.assertEqual(DataMapper.map_age_to_group("40"), 3)
        self.assertEqual(DataMapper.map_age_to_group("60"), 4)
        self.assertEqual(DataMapper.map_age_to_group("invalid"), 2)

    def test_model_type_mapping_preserves_human_default(self):
        self.assertEqual(DataMapper.map_model_type_to_db("character"), 1)
        self.assertEqual(DataMapper.map_model_type_to_db("objects"), 3)
        self.assertEqual(DataMapper.map_model_type_to_db(None), 2)

    def test_character_data_uses_mapper_and_placeholder_values(self):
        self.assertEqual(
            create_character_data(None, None, "20", "남성", "차분함", "INTJ"),
            {
                "name": "미지정",
                "description": "미지정",
                "age_range": "20대",
                "gender": "MALE",
                "personality": "차분함",
                "mbti": "INTJ",
            },
        )


if __name__ == "__main__":
    unittest.main()
