from typing import Dict, List, Optional, Union, Iterable

from pydantic import BaseModel, Field

# TODO __all__

class PropertyBaseModel(BaseModel):
    """
    Workaround for serializing properties with pydantic until
    https://github.com/samuelcolvin/pydantic/issues/935
    is solved
    """
    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if isinstance(getattr(cls, prop), property) and prop not in ("__values__", "fields")]

    def dict(
        self,
        *,
        include: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        exclude: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> 'DictStrAny':
        attribs = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none
        )
        props = self.get_properties()
        # Include and exclude properties
        if include:
            props = [prop for prop in props if prop in include]
        if exclude:
            props = [prop for prop in props if prop not in exclude]

        # Update the attribute dict with the properties
        if props:
            attribs.update({prop: getattr(self, prop) for prop in props})

        return attribs

class DenseSparseOps(PropertyBaseModel):
    """
    Pydnatic model for num_dense_ops and num_sparse_ops
    """
    dense: int = Field(
        description="TODO"
    )
    sparse: int = Field(
        description="TODO"
    )
    @property
    def sparsity(self):
        total_ops = self.sparse + self.dense
        if total_ops > 0:
            return self.sparse / (self.sparse + self.dense)
        else:
            return 0

class Operations(BaseModel):
    """
    Pydnatic model for describing the operation counts of a node or model
    """
    num_operations: DenseSparseOps = Field(
        description="TODO: FP + Q"
    )
    multiply_accumulates: DenseSparseOps = Field(
        description="TODO"
    )

class ModelOperations(BaseModel):
    """
    TODO
    """
    floating_or_quantized_ops: DenseSparseOps = Field(
        description="TODO: FP + Q"
    )
    floating_point_ops: DenseSparseOps = Field(
        description="TODO: FP + Q"
    )
    quantized_ops: DenseSparseOps = Field(
        description="TODO: FP + Q"
    )
    multiply_accumulates: DenseSparseOps = Field(
        description="TODO"
    )

class DenseSparseValues(PropertyBaseModel):
    """
    TODO
    """
    num_non_zero: int = Field(
        description="TODO"
    )
    num_zero: int = Field(
        description="TODO"
    )

    @property
    def sparsity(self):
        total_values = self.num_non_zero + self.num_zero
        if total_values > 0:
            return self.num_zero / total_values
        else:
            return 0

class Parameters(BaseModel):
    """
    TODO
    """
    single: DenseSparseValues = Field(
        description="TODO"
    )
    four_block: DenseSparseValues = Field(
        description="TODO"
    )

class WeightAnalysis(BaseModel):
    """
    Pydantic model for the weight of a node
    """

    name: Optional[str] = Field(description="The name of the weight")
    shape: Optional[List[Union[None, int]]] = Field(
        description="The weight's shape (assuming a batch size of 1)"
    )
    parameters: Parameters = Field(
        description="TODO"
    )
    operations: Operations = Field(
        description="TODO"
    )
    dtype: str = Field(description="The weight's data type")

class BiasAnalysis(BaseModel):
    """
    Pydantic model for the weight of a node
    """

    shape: Optional[List[Union[None, int]]] = Field(
        description="The bias' shape (assuming a batch size of 1)"
    )
    operations: Operations = Field(
        description="TODO"
    )
    dtype: str = Field(description="The bias' data type")

class NodeIO(BaseModel):
    """
    Pydantic model for the inputs and outputs of a node in the onnx model graph
    """

    name: str = Field(description="Name of the input/output in onnx model graph")
    shape: Optional[List[Union[None, int]]] = Field(
        description="Shape of the input/output in onnx model graph (assuming a "
        "batch size of 1)"
    )
    dtype: Optional[str] = Field(
        description="Data type of the values from the input/output"
    )
