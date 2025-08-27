from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class IdentityEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):

    def fit(self, data):
        """Fit the encoder to the data.

        Args:
            data: The input data to fit.

        Returns:
            self: The fitted encoder.
        """
        self.data_type_ = type(data)
        return self

    def fit_transform(self, data):
        """Fit the encoder to the data and transform it.

        Args:
            data: The input data to fit and transform.

        Returns:
            transformed_data: The transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        """Transform the input data.

        Args:
            data: The input data to transform.

        Returns:
            transformed_data: The transformed data.
        """
        check_is_fitted(self)

        return data

    def inverse_transform(self, data):
        """Inverse transform the input data.

        Args:
            data: The input data to inverse transform.

        Returns:
            transformed_data: The transformed data.
        """
        check_is_fitted(self)

        return data

    def _more_tags(self):
        tags = super().__sklearn_tags__()

        return tags
