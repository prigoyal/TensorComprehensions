Semantics
=========

Types
-----

Values between statements are always tensors of primitive types (e.g. :code:`float(A,B)`, a tensor of rank 2).
They can be 0-rank and omit the dimension list (e.g :code:`float`).
Size variables (e.g. :code:`A` and :code:`B` in :code:`float(A,B)`) are used to represent the sizes of the dimensions.
If a size variable is repeated it means that tensors of that type must share the same size in that dimension.
Size variables evaluate to the size of the dimension when used in expressions.

The type of output values is omitted and is inferred based on how it is defined as described below.

Data Layout
-----------
The memory layout implied by TC is row-major (C-like).

Variable Scoping
----------------

There are three different kinds of variables, which all share the same namespace:

1. size variables, introduced by tensor types in the type signature, which evaluate to the size of the dimension;
2. tensor variables, introduced by tensor types in the type signature, with ranges either prescribed (input tensors) or inferred (output tensors);
3. loop index variables, are implicitly defined when used in a statement.

When an identifier is used in a statement but is otherwise not in scope, it is defined to be an index variable for that statement.
Each index variable has an associated range :code:`[b,e)` over which it operates.
That range is inferred by its use, as described below.
Index variables go out of scope after the statement, allowing the reuse of short variable names like :code:`i`.

Implied Reductions
------------------

If an index variable appears on the right but not on the left of a statement,
it is a reduction index for the statement.
If a statement has one or more reduction variables then it must specify a :code:`reduction`
operator such as :code:`+` or :code:`max`.
There is only one reduction operator for the entire statement because
combinations like max/+ on different dimensions have different mathematical meanings depending on loop order.
All reduction operators are considered to be associative and commutative to
allow for arbitrary order of evaluation.

Size Expressions
----------------

Size expressions are a subset of normal expressions that can be used in explicit range constraints and in pattern matching.
They are any expression over integral scalars that do not include tensor reads :code:`T(...)` or any loop index variables.
They may include size variables, or dimension specifiers :code:`T.1`, for tensors that have already been defined in previous statements.
These values can be computed without performing any tensor-wide loops.

Statements
----------

A statement specifies a new operation to define, an optional reduction, and a right hand side::

    v(index_variables) reduction=! rhs_expression

:code:`index_variabes` must be a list of index variables defined in the :code:`rhs_expressions`
:code:`reduction` is optional if all index variables appear on the left hand side.
The value computed for tensor :code:`v` is equivalent to first assigning all
elements of :code:`v` to the identity value of :code:`reduction`, then
evaluating :code:`rhs_expression` at all points in the iteration space defined
by the ranges of the loop index variables and reducing into the entry of the
tensor specified on the left-hand side. The order in which these expressions
are evaluated should not change the result because the reduction is
associative and commutative.

Expressions
-----------

Mathematical expressions behave as expected, including built-in functions like :code:`log(...)`.

:code:`tensor_variable(exp_list)` represents a read of a tensor at the indices defined by evaluating :code:`exp_list`. :code:`exp_list` can include arbitrary expressions (pattern matching of indices is limited to linear expressions, but actual computation is not). The effect of reading outside of the valid range of the tensor results in undefined behavior.

Grammar
-------

The EBNF for the TC comprehension language is::

    number ::= <C's number literal formal>
    ident ::= [_a-zA-Z][_a-zA-Z0-9]*
    exp ::= number
          | exp '+' exp
          | exp '-' exp
          | exp '* ' exp
          | exp '/' exp
          | '-' exp
          | a '?' 'b' : 'c'
          | ...  # other standard numeric built-ins
          | ident '.' number # size of the number'th dimension of ident
          | ident'('exp_list')' # built-in functions, and tensor access

    reduction ::= '+' | '* ' | 'min' | 'max' | <other associative reductions>
    range_constraint ::= ident '=' '[' exp ',' exp '['
    stmt ::= ident(ident_list) (reduction)? '=' exp [ 'where' range_constraint_list ]
           | indent_list = ident(ident_list) # call another TC function

    param ::= type ident
            | ident
    return ::= param
             | ident # return type inferred from file

    scalar_type = 'float' | 'double' | 'long' | 'byte' | ...

    # eventually maybe parametric polymorphism
    type ::= scalar_type ['(' ident_list ')']

    func ::= 'def' ident '(' param_list ')' '->' '(' return_list ')' '{'
                stmt_list
             '}'

    exp_list ::= <comma separated exp list>
    ident_list ::= <comma separated ident list>
    param_list ::= <comma separated param list>
    stmt_list ::= <whitespace separated stmt list>
    return_list ::= <comma separated reutnr list>
    range_constraint ::= <non-empty comma separated list>
