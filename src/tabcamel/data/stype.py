from enum import Enum


class stype(Enum):
    r"""The semantic type of a column.

    A semantic type denotes the semantic meaning of a column, and denotes how
    columns are encoded into an embedding space within tabular deep learning
    models.

    """

    numerical          = "numerical"
    categorical        = "categorical"
    text_embedded      = "text_embedded"
    text_tokenized     = "text_tokenized"
    multicategorical   = "multicategorical"
    sequence_numerical = "sequence_numerical"
    timestamp          = "timestamp"
    image_embedded     = "image_embedded"
    embedding          = "embedding"

    @property
    def is_text_stype(self) -> bool:
        return self in [stype.text_embedded, stype.text_tokenized]

    @property
    def is_image_stype(self) -> bool:
        return self in [stype.image_embedded]

    @property
    def use_multi_nested_tensor(self) -> bool:
        return self in [stype.multicategorical, self.sequence_numerical]

    @property
    def use_multi_embedding_tensor(self) -> bool:
        return self in [stype.text_embedded, stype.image_embedded, stype.embedding]

    @property
    def use_dict_multi_nested_tensor(self) -> bool:
        return self in [stype.text_tokenized]

    @property
    def use_multi_tensor(self) -> bool:
        return self.use_multi_nested_tensor or self.use_multi_embedding_tensor

    @property
    def parent(self):
        if self == stype.text_embedded:
            return stype.embedding
        elif self == stype.image_embedded:
            return stype.embedding
        else:
            return self

    @property
    def pandas_dtype(self):
        match self:
            case stype.numerical:
                return "float64"
            case stype.categorical:
                return "string"
            case _:
                raise ValueError(f"Unsupported pandas dtype for {self}")

    @property
    def sdmetrics_dtype(self):
        match self:
            case stype.numerical:
                return {
                    "sdtype": "numerical",
                    "compute_representation": "Float",
                }
            case stype.categorical:
                return {
                    "sdtype": "categorical",
                }

    def __str__(self) -> str:
        return f"{self.name}"


numerical          = stype("numerical")
categorical        = stype("categorical")
text_embedded      = stype("text_embedded")
text_tokenized     = stype("text_tokenized")
multicategorical   = stype("multicategorical")
sequence_numerical = stype("sequence_numerical")
timestamp          = stype("timestamp")
image_embedded     = stype("image_embedded")
embedding          = stype("embedding")
