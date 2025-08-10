from __future__ import annotations

import copy
from abc import abstractmethod

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder, QuantileTransformer,
                                   StandardScaler)


class BaseTransform:
    """Abstract base class for all data transformations.

    This class provides the foundation for implementing data preprocessing
    transformations that can be fitted on training data and applied to
    new data. All transformations follow the scikit-learn transformer pattern.

    The class ensures proper state management and provides a consistent
    interface for all transformation operations including fitting, transforming,
    and inverse transforming data.

    Attributes:
        is_fitted (bool): Whether the transformer has been fitted to data

    Note:
        This is an abstract class and cannot be instantiated directly.
        Subclasses must implement _fit, _transform, and _inverse_transform methods.
    """

    def __init__(self):
        """Initialize the BaseTransform class.

        Sets up the basic state tracking for the transformer.
        """
        super().__init__()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the transform is already fitted."""
        return self._is_fitted

    def fit(
        self,
        data_df: pd.DataFrame,
    ):
        """Fit the transformer to the provided data.

        This method learns the parameters needed for transformation from the
        training data. It must be called before transform() can be used.

        Args:
            data_df (pd.DataFrame): Training data to fit the transformer on.
                The DataFrame should contain the features that will be transformed.

        Note:
            After calling this method, is_fitted will be set to True.
            The actual fitting logic is implemented in the _fit() method
            of each subclass.
        """
        self._fit(data_df)
        self._is_fitted = True

    def transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform the provided data using the fitted parameters.

        Applies the transformation learned during fitting to new data.
        The transformer must be fitted before this method can be called.

        Args:
            data_df (pd.DataFrame): Data to transform. Should have the same
                structure as the data used for fitting.

        Returns:
            pd.DataFrame: Transformed data with the same index as input but
                potentially different columns depending on the transformation.

        Raises:
            ValueError: If the transformer has not been fitted yet.

        Example:
            >>> transformer = SomeTransform()
            >>> transformer.fit(train_data)
            >>> transformed_data = transformer.transform(test_data)
        """
        if not self.is_fitted:
            raise ValueError(
                f"'{self.__class__.__name__}' is not yet fitted ."
                f"Please run `fit()` first before attempting to "
                f"transform the DataFrame."
            )
        data_df_transformed = self._transform(copy.deepcopy(data_df))
        return data_df_transformed

    def inverse_transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Inverse transforms the data.

        Args:
            data_df (pd.DataFrame): Data to inverse transform.

        Raises:
            ValueError: If the transform is not yet fitted.

        Returns:
            pd.DataFrame: Inverse transformed data.
        """
        if not self.is_fitted:
            raise ValueError(
                f"'{self.__class__.__name__}' is not yet fitted ."
                f"Please run `fit()` first before attempting to "
                f"inverse transform the DataFrame."
            )
        data_df_inverse_transformed = self._inverse_transform(copy.deepcopy(data_df))

        return data_df_inverse_transformed

    @abstractmethod
    def _fit(
        self,
        data_df: pd.DataFrame,
    ):
        """Fits the transform to the data.

        Args:
            data_df (pd.DataFrame): Data to fit the transform to.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transforms the data.

        Args:
            data_df (pd.DataFrame): Data to transform.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            pd.DataFrame: Transformed data.
        """
        raise NotImplementedError

    @abstractmethod
    def _inverse_transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Inverse transforms the data.

        Args:
            data_df (pd.DataFrame): Data to inverse transform.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            pd.DataFrame: Inverse transformed data.
        """
        raise NotImplementedError


class SimpleImputeTransform(BaseTransform):

    def __init__(
        self,
        categorical_feature_list: list,
        numerical_feature_list: list,
        strategy_categorical: str,
        strategy_numerical: str,
        missing_values=pd.NA,
        fill_value=None,
        copy=False,
        add_indicator=False,
        keep_empty_features=False,
    ):
        """Initialises the SimpleImputeTransform class.
        Args:
            categorical_feature_list (list): List of categorical feature names.
            numerical_feature_list (list): List of numerical feature names.
            strategy_categorical (str): Strategy for imputing categorical features.
            strategy_numerical (str): Strategy for imputing numerical features.
            missing_values: The placeholder for the missing values.
            fill_value: The value to replace missing values with.
            copy (bool): Whether to copy the data before transforming.
            As we have done deepcopy in BaseTransform, this is set to False by default to avoid unnecessary copies.
            add_indicator (bool): Whether to add an indicator for missing values.
            keep_empty_features (bool): Whether to keep empty features in the output.
        """
        super().__init__()

        # === Basic configurations ===
        self.categorical_feature_list = categorical_feature_list
        self.numerical_feature_list = numerical_feature_list

        # === Set the imputers for categorical and numerical features ===
        self._imputer_categorical = SimpleImputer(
            missing_values=missing_values,
            strategy=strategy_categorical,
            fill_value=fill_value,
            copy=copy,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )
        self._imputer_numerical = SimpleImputer(
            missing_values=missing_values,
            strategy=strategy_numerical,
            fill_value=fill_value,
            copy=copy,
            add_indicator=add_indicator,
            keep_empty_features=keep_empty_features,
        )

    def _fit(
        self,
        data_df: pd.DataFrame,
    ):
        """Fits the transform to the data.

        Args:
            data_df (pd.DataFrame): Data to fit the transform to.
        """
        if self.categorical_feature_list:
            self._imputer_categorical.fit(data_df[self.categorical_feature_list])
        if self.numerical_feature_list:
            self._imputer_numerical.fit(data_df[self.numerical_feature_list])

    def _transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transforms the data.

        Args:
            data_df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data.
        """
        data_df_transformed = data_df

        if self.categorical_feature_list:
            data_df_transformed[self.categorical_feature_list] = self._imputer_categorical.transform(
                data_df[self.categorical_feature_list]
            )
        if self.numerical_feature_list:
            data_df_transformed[self.numerical_feature_list] = self._imputer_numerical.transform(
                data_df[self.numerical_feature_list]
            )

        return data_df_transformed

    def _inverse_transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Inverse transforms the data.
        Note that we simply return the original data as "add_indicator" is not fully supported yet.

        Args:
            data_df (pd.DataFrame): Data to inverse transform.

        Returns:
            pd.DataFrame: Inverse transformed data.
        """
        data_df_inverse_transformed = data_df

        return data_df_inverse_transformed


class NumericTransform(BaseTransform):

    def __init__(
        self,
        numerical_feature_list: list,
        strategy: str,
        include_categorical: bool,
        train_num_samples: int = None,
    ):
        """Initialises the NumericTransform class."""
        super().__init__()

        self.numerical_feature_list = numerical_feature_list
        self.include_categorical = include_categorical
        self.train_num_samples = train_num_samples

        match strategy:
            case "standard":
                self._scaler = StandardScaler()
            case "minmax":
                self._scaler = MinMaxScaler()
            case "quantile":
                if train_num_samples is None:
                    raise ValueError(f"train_num_samples must be provided for {strategy} strategy.")
                # The settings are consistent with TabZilla: https://github.com/naszilla/tabzilla/blob/main/TabZilla/tabzilla_data_processing.py#L134
                self._scaler = QuantileTransformer(n_quantiles=min(self.train_num_samples, 1000))

    def _fit(
        self,
        data_df: pd.DataFrame,
    ):
        """Fits the transform to the data.

        Args:
            data_df (pd.DataFrame): Data to fit the transform to.
        """
        if self.include_categorical:
            self._scaler.fit(data_df)
        elif len(self.numerical_feature_list) > 0:
            self._scaler.fit(data_df[self.numerical_feature_list])
        else:
            # Numerical transforms cannot work when there is no numerical features.
            self._scaler = None

    def _transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transforms the data.

        Args:
            data_df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data.
        """
        if self._scaler is None:
            return data_df

        if self.include_categorical:
            data_df_transformed = self._scaler.transform(data_df)
        else:
            data_df_transformed = data_df
            data_df_transformed[self.numerical_feature_list] = self._scaler.transform(
                data_df[self.numerical_feature_list]
            )

        return pd.DataFrame(data_df_transformed, columns=data_df.columns, index=data_df.index)

    def _inverse_transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Inverse transforms the data.

        Args:
            data_df (pd.DataFrame): Data to inverse transform.

        Returns:
            pd.DataFrame: Inverse transformed data.
        """
        if self._scaler is None:
            return data_df

        if self.include_categorical:
            data_df_inverse_transformed = self._scaler.inverse_transform(data_df)
        else:
            data_df_inverse_transformed = data_df
            data_df_inverse_transformed[self.numerical_feature_list] = self._scaler.inverse_transform(
                data_df[self.numerical_feature_list]
            )

        return pd.DataFrame(data_df_inverse_transformed, columns=data_df.columns, index=data_df.index)


class CategoryTransform(BaseTransform):

    def __init__(
        self,
        categorical_feature_list: list,
        strategy: str,
    ):
        super().__init__()

        self.categorical_feature_list = categorical_feature_list
        self.strategy = strategy

        match strategy:
            case "onehot":
                # ncode categorical features as a one-hot numeric array.
                self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            case "ordinal":
                # This results in a single column of integers (0 to n_categories - 1) per feature.
                self._encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            case _:
                raise ValueError(f"Invalid strategy '{strategy}'.")

    def _fit(
        self,
        data_df: pd.DataFrame,
    ):
        self._original_column_list = data_df.columns.tolist()
        if len(self.categorical_feature_list) > 0:
            self._encoder.fit(data_df[self.categorical_feature_list])
        else:
            # Categorical transforms cannot work when there is no categorical features.
            self._encoder = None

    def _transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._encoder is None:
            return data_df

        cat_df_transformed = self._encoder.transform(data_df[self.categorical_feature_list])
        cat_df_transformed = pd.DataFrame(
            cat_df_transformed,
            columns=self._encoder.get_feature_names_out(self.categorical_feature_list),
            index=data_df.index,
        )

        data_df_transformed = data_df.drop(self.categorical_feature_list, axis=1)
        data_df_transformed = pd.concat([data_df_transformed, cat_df_transformed], axis=1)

        return data_df_transformed

    def _inverse_transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._encoder is None:
            # If the encoder does not exist, return the original data.
            return data_df

        """One-hot encoder can inverse transform non-one-hot encoded data.
        from sklearn.preprocessing import OneHotEncoder

        enc = OneHotEncoder(handle_unknown="ignore")
        X = [["Male", 1], ["Female", 3], ["Female", 2]]
        print(enc.fit(X))
        print(enc.categories_)
        print(enc.transform([["Female", 1], ["Male", 4]]).toarray())
        print(
            enc.inverse_transform(
                [
                    [0, 1, 1, 0, 0],  # Male, 1 -> One-hot transform would sort the categories
                    [0, 0, 0, 1, 0],  # None, 2
                    [0, 0, 0, 0, 0],  # None, None
                    [0.3, 0.7, 0.2, 0.7, 0.1],  # Male, 2
                    [0.7, 0.3, 0.2, 0.7, 0.1],  # Female, 2
                    [1, 1, 0.3, 0.3, 0.3],  # Female, 1
                ]
            )
        )
        print(enc.get_feature_names_out(["gender", "group"]))
        """
        categorical_feature_list_encoded = self._encoder.get_feature_names_out(self.categorical_feature_list)
        cat_df = data_df[categorical_feature_list_encoded]
        cat_df_inverse_transformed = self._encoder.inverse_transform(cat_df)
        cat_df_inverse_transformed = pd.DataFrame(
            cat_df_inverse_transformed,
            columns=self.categorical_feature_list,
            index=data_df.index,
        )

        data_df_inverse_transformed = data_df
        data_df_inverse_transformed = data_df_inverse_transformed.drop(categorical_feature_list_encoded, axis=1)
        data_df_inverse_transformed = pd.concat([data_df_inverse_transformed, cat_df_inverse_transformed], axis=1)
        data_df_inverse_transformed = data_df_inverse_transformed[self._original_column_list]

        return data_df_inverse_transformed

    @property
    def categories_(self) -> list:
        return self._encoder.categories_


class TargetTransform(BaseTransform):

    def __init__(
        self,
        task: str,
        target_feature: str,
        copy: bool = False,
        with_mean: bool = True,
        with_std: bool = True,
    ):
        super().__init__()

        self.task = task
        self.target_feature = target_feature

        match task:
            case "classification":
                # The labels are sorted in alphabetic order before encoding.
                self._encoder = LabelEncoder()
            case "regression":
                self._encoder = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
            case _:
                raise ValueError(f"Invalid task '{task}'.")

    def _fit(
        self,
        data_df: pd.DataFrame,
    ):
        self._encoder.fit(data_df[[self.target_feature]])

    def _transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        data_df_transformed = data_df
        data_df_transformed[self.target_feature] = self._encoder.transform(data_df[[self.target_feature]])

        return data_df_transformed

    def _inverse_transform(
        self,
        data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        data_df_inverse_transformed = data_df
        data_df_inverse_transformed[self.target_feature] = self._encoder.inverse_transform(
            data_df[[self.target_feature]]
        )

        return data_df_inverse_transformed

    @property
    def encoded2class(self) -> dict:
        if self.task == "classification":
            return {i: c for i, c in enumerate(self._encoder.classes_)}

        return None
