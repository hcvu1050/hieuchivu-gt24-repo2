��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
�
Adam/v/output_NN/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/v/output_NN/bias
{
)Adam/v/output_NN/bias/Read/ReadVariableOpReadVariableOpAdam/v/output_NN/bias*
_output_shapes
:*
dtype0
�
Adam/m/output_NN/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/m/output_NN/bias
{
)Adam/m/output_NN/bias/Read/ReadVariableOpReadVariableOpAdam/m/output_NN/bias*
_output_shapes
:*
dtype0
�
Adam/v/output_NN/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/v/output_NN/kernel
�
+Adam/v/output_NN/kernel/Read/ReadVariableOpReadVariableOpAdam/v/output_NN/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/output_NN/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/m/output_NN/kernel
�
+Adam/m/output_NN/kernel/Read/ReadVariableOpReadVariableOpAdam/m/output_NN/kernel*
_output_shapes

:@*
dtype0
~
Adam/v/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_9/bias
w
'Adam/v/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_9/bias
w
'Adam/m/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/v/dense_9/kernel

)Adam/v/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/m/dense_9/kernel

)Adam/m/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/kernel*
_output_shapes

:@@*
dtype0
~
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes

:@@*
dtype0
~
Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_7/bias
w
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_7/bias
w
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/v/dense_7/kernel

)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/m/dense_7/kernel

)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes

:@@*
dtype0
~
Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_6/bias
w
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_6/bias
w
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/v/dense_6/kernel

)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/m/dense_6/kernel

)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel*
_output_shapes

:@@*
dtype0
~
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:@*
dtype0
~
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/v/dense_5/kernel
�
)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*&
shared_nameAdam/m/dense_5/kernel
�
)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/v/output_NN/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/v/output_NN/bias_1

+Adam/v/output_NN/bias_1/Read/ReadVariableOpReadVariableOpAdam/v/output_NN/bias_1*
_output_shapes
:*
dtype0
�
Adam/m/output_NN/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/m/output_NN/bias_1

+Adam/m/output_NN/bias_1/Read/ReadVariableOpReadVariableOpAdam/m/output_NN/bias_1*
_output_shapes
:*
dtype0
�
Adam/v/output_NN/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/v/output_NN/kernel_1
�
-Adam/v/output_NN/kernel_1/Read/ReadVariableOpReadVariableOpAdam/v/output_NN/kernel_1*
_output_shapes

: *
dtype0
�
Adam/m/output_NN/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_nameAdam/m/output_NN/kernel_1
�
-Adam/m/output_NN/kernel_1/Read/ReadVariableOpReadVariableOpAdam/m/output_NN/kernel_1*
_output_shapes

: *
dtype0
~
Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_4/bias
w
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_4/bias
w
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/dense_4/kernel

)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/dense_4/kernel

)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel*
_output_shapes

:  *
dtype0
~
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/dense_3/kernel

)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/dense_3/kernel

)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes

:  *
dtype0
~
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_2/bias
w
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_2/bias
w
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/dense_2/kernel

)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/dense_2/kernel

)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel*
_output_shapes

:  *
dtype0
~
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
: *
dtype0
~
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/v/dense_1/kernel

)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/m/dense_1/kernel

)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes

:  *
dtype0
z
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/dense/bias
s
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes
: *
dtype0
z
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/dense/bias
s
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	� *
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	� *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
t
output_NN/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput_NN/bias
m
"output_NN/bias/Read/ReadVariableOpReadVariableOpoutput_NN/bias*
_output_shapes
:*
dtype0
|
output_NN/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameoutput_NN/kernel
u
$output_NN/kernel/Read/ReadVariableOpReadVariableOpoutput_NN/kernel*
_output_shapes

:@*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:@*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:@@*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:@*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:@@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:@*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:@@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:@*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	�@*
dtype0
x
output_NN/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameoutput_NN/bias_1
q
$output_NN/bias_1/Read/ReadVariableOpReadVariableOpoutput_NN/bias_1*
_output_shapes
:*
dtype0
�
output_NN/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *#
shared_nameoutput_NN/kernel_1
y
&output_NN/kernel_1/Read/ReadVariableOpReadVariableOpoutput_NN/kernel_1*
_output_shapes

: *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:  *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:  *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:  *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:  *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	� *
dtype0
�
serving_default_input_GroupPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
serving_default_input_TechniquePlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_Groupserving_default_input_Techniquedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_NN/kernel_1output_NN/bias_1dense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasoutput_NN/kerneloutput_NN/bias*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_779007

NoOpNoOp
ے
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
Group_NN
	Technique_NN

dot_product
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 
�
2layer_with_weights-0
2layer-0
3layer_with_weights-1
3layer-1
4layer_with_weights-2
4layer-2
5layer_with_weights-3
5layer-3
6layer_with_weights-4
6layer-4
7layer_with_weights-5
7layer-5
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
�
>layer_with_weights-0
>layer-0
?layer_with_weights-1
?layer-1
@layer_with_weights-2
@layer-2
Alayer_with_weights-3
Alayer-3
Blayer_with_weights-4
Blayer-4
Clayer_with_weights-5
Clayer-5
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla*

Wserving_default* 
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEoutput_NN/kernel_1'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEoutput_NN/bias_1'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_5/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_5/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_6/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_6/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_7/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_7/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_8/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_8/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_9/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_9/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEoutput_NN/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEoutput_NN/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*

X0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

kernel
bias*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kernel
bias*
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

kernel
bias*
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

kernel
bias*
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

kernel
bias*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
 bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

!kernel
"bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

#kernel
$bias*
Z
0
1
2
3
4
5
6
 7
!8
"9
#10
$11*
Z
0
1
2
3
4
5
6
 7
!8
"9
#10
$11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
Q0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23*
* 
* 
<
�	variables
�	keras_api

�total

�count*

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
.
20
31
42
53
64
75*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
 1*

0
 1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

!0
"1*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
.
>0
?1
@2
A3
B4
C5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
^X
VARIABLE_VALUEAdam/m/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_4/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_4/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_4/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/output_NN/kernel_12optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/output_NN/kernel_12optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/output_NN/bias_12optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/output_NN/bias_12optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_5/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_5/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_5/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_5/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_6/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_6/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_6/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_6/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_7/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_7/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_7/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_7/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_8/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_9/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_9/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/output_NN/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/output_NN/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/output_NN/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/output_NN/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_NN/kernel_1output_NN/bias_1dense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasoutput_NN/kerneloutput_NN/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/output_NN/kernel_1Adam/v/output_NN/kernel_1Adam/m/output_NN/bias_1Adam/v/output_NN/bias_1Adam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biasAdam/m/output_NN/kernelAdam/v/output_NN/kernelAdam/m/output_NN/biasAdam/v/output_NN/biastotalcountConst*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_780353
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_NN/kernel_1output_NN/bias_1dense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasoutput_NN/kerneloutput_NN/bias	iterationlearning_rateAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/output_NN/kernel_1Adam/v/output_NN/kernel_1Adam/m/output_NN/bias_1Adam/v/output_NN/bias_1Adam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biasAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biasAdam/m/output_NN/kernelAdam/v/output_NN/kernelAdam/m/output_NN/biasAdam/v/output_NN/biastotalcount*X
TinQ
O2M*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_780591��
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_777694

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_model1_layer_call_fn_778834
input_group
input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_groupinput_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model1_layer_call_and_return_conditional_losses_778783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:YU
(
_output_shapes
:����������
)
_user_specified_nameinput_Technique:U Q
(
_output_shapes
:����������
%
_user_specified_nameinput_Group
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_779656

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�'
�
B__inference_model1_layer_call_and_return_conditional_losses_778659

inputs
inputs_1"
group_nn_778593:	� 
group_nn_778595: !
group_nn_778597:  
group_nn_778599: !
group_nn_778601:  
group_nn_778603: !
group_nn_778605:  
group_nn_778607: !
group_nn_778609:  
group_nn_778611: !
group_nn_778613: 
group_nn_778615:&
technique_nn_778618:	�@!
technique_nn_778620:@%
technique_nn_778622:@@!
technique_nn_778624:@%
technique_nn_778626:@@!
technique_nn_778628:@%
technique_nn_778630:@@!
technique_nn_778632:@%
technique_nn_778634:@@!
technique_nn_778636:@%
technique_nn_778638:@!
technique_nn_778640:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinputsgroup_nn_778593group_nn_778595group_nn_778597group_nn_778599group_nn_778601group_nn_778603group_nn_778605group_nn_778607group_nn_778609group_nn_778611group_nn_778613group_nn_778615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777839�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinputs_1technique_nn_778618technique_nn_778620technique_nn_778622technique_nn_778624technique_nn_778626technique_nn_778628technique_nn_778630technique_nn_778632technique_nn_778634technique_nn_778636technique_nn_778638technique_nn_778640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778223z
l2_normalize/SquareSquare)Group_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalizeMul)Group_NN/StatefulPartitionedCall:output:0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
l2_normalize_1/SquareSquare-Technique_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalize_1Mul-Technique_NN/StatefulPartitionedCall:output:0l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:����������
dot/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_778512k
IdentityIdentitydot/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 Group_NN/StatefulPartitionedCall Group_NN/StatefulPartitionedCall2L
$Technique_NN/StatefulPartitionedCall$Technique_NN/StatefulPartitionedCall:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�D
__inference__traced_save_780353
file_prefix6
#read_disablecopyonread_dense_kernel:	� 1
#read_1_disablecopyonread_dense_bias: 9
'read_2_disablecopyonread_dense_1_kernel:  3
%read_3_disablecopyonread_dense_1_bias: 9
'read_4_disablecopyonread_dense_2_kernel:  3
%read_5_disablecopyonread_dense_2_bias: 9
'read_6_disablecopyonread_dense_3_kernel:  3
%read_7_disablecopyonread_dense_3_bias: 9
'read_8_disablecopyonread_dense_4_kernel:  3
%read_9_disablecopyonread_dense_4_bias: >
,read_10_disablecopyonread_output_nn_kernel_1: 8
*read_11_disablecopyonread_output_nn_bias_1:;
(read_12_disablecopyonread_dense_5_kernel:	�@4
&read_13_disablecopyonread_dense_5_bias:@:
(read_14_disablecopyonread_dense_6_kernel:@@4
&read_15_disablecopyonread_dense_6_bias:@:
(read_16_disablecopyonread_dense_7_kernel:@@4
&read_17_disablecopyonread_dense_7_bias:@:
(read_18_disablecopyonread_dense_8_kernel:@@4
&read_19_disablecopyonread_dense_8_bias:@:
(read_20_disablecopyonread_dense_9_kernel:@@4
&read_21_disablecopyonread_dense_9_bias:@<
*read_22_disablecopyonread_output_nn_kernel:@6
(read_23_disablecopyonread_output_nn_bias:-
#read_24_disablecopyonread_iteration:	 1
'read_25_disablecopyonread_learning_rate: @
-read_26_disablecopyonread_adam_m_dense_kernel:	� @
-read_27_disablecopyonread_adam_v_dense_kernel:	� 9
+read_28_disablecopyonread_adam_m_dense_bias: 9
+read_29_disablecopyonread_adam_v_dense_bias: A
/read_30_disablecopyonread_adam_m_dense_1_kernel:  A
/read_31_disablecopyonread_adam_v_dense_1_kernel:  ;
-read_32_disablecopyonread_adam_m_dense_1_bias: ;
-read_33_disablecopyonread_adam_v_dense_1_bias: A
/read_34_disablecopyonread_adam_m_dense_2_kernel:  A
/read_35_disablecopyonread_adam_v_dense_2_kernel:  ;
-read_36_disablecopyonread_adam_m_dense_2_bias: ;
-read_37_disablecopyonread_adam_v_dense_2_bias: A
/read_38_disablecopyonread_adam_m_dense_3_kernel:  A
/read_39_disablecopyonread_adam_v_dense_3_kernel:  ;
-read_40_disablecopyonread_adam_m_dense_3_bias: ;
-read_41_disablecopyonread_adam_v_dense_3_bias: A
/read_42_disablecopyonread_adam_m_dense_4_kernel:  A
/read_43_disablecopyonread_adam_v_dense_4_kernel:  ;
-read_44_disablecopyonread_adam_m_dense_4_bias: ;
-read_45_disablecopyonread_adam_v_dense_4_bias: E
3read_46_disablecopyonread_adam_m_output_nn_kernel_1: E
3read_47_disablecopyonread_adam_v_output_nn_kernel_1: ?
1read_48_disablecopyonread_adam_m_output_nn_bias_1:?
1read_49_disablecopyonread_adam_v_output_nn_bias_1:B
/read_50_disablecopyonread_adam_m_dense_5_kernel:	�@B
/read_51_disablecopyonread_adam_v_dense_5_kernel:	�@;
-read_52_disablecopyonread_adam_m_dense_5_bias:@;
-read_53_disablecopyonread_adam_v_dense_5_bias:@A
/read_54_disablecopyonread_adam_m_dense_6_kernel:@@A
/read_55_disablecopyonread_adam_v_dense_6_kernel:@@;
-read_56_disablecopyonread_adam_m_dense_6_bias:@;
-read_57_disablecopyonread_adam_v_dense_6_bias:@A
/read_58_disablecopyonread_adam_m_dense_7_kernel:@@A
/read_59_disablecopyonread_adam_v_dense_7_kernel:@@;
-read_60_disablecopyonread_adam_m_dense_7_bias:@;
-read_61_disablecopyonread_adam_v_dense_7_bias:@A
/read_62_disablecopyonread_adam_m_dense_8_kernel:@@A
/read_63_disablecopyonread_adam_v_dense_8_kernel:@@;
-read_64_disablecopyonread_adam_m_dense_8_bias:@;
-read_65_disablecopyonread_adam_v_dense_8_bias:@A
/read_66_disablecopyonread_adam_m_dense_9_kernel:@@A
/read_67_disablecopyonread_adam_v_dense_9_kernel:@@;
-read_68_disablecopyonread_adam_m_dense_9_bias:@;
-read_69_disablecopyonread_adam_v_dense_9_bias:@C
1read_70_disablecopyonread_adam_m_output_nn_kernel:@C
1read_71_disablecopyonread_adam_v_output_nn_kernel:@=
/read_72_disablecopyonread_adam_m_output_nn_bias:=
/read_73_disablecopyonread_adam_v_output_nn_bias:)
read_74_disablecopyonread_total: )
read_75_disablecopyonread_count: 
savev2_const
identity_153��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	� w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: {
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_4_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:  y
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_4_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead,read_10_disablecopyonread_output_nn_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp,read_10_disablecopyonread_output_nn_kernel_1^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

: 
Read_11/DisableCopyOnReadDisableCopyOnRead*read_11_disablecopyonread_output_nn_bias_1"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp*read_11_disablecopyonread_output_nn_bias_1^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_5_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_5_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_dense_6_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@@{
Read_15/DisableCopyOnReadDisableCopyOnRead&read_15_disablecopyonread_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp&read_15_disablecopyonread_dense_6_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_16/DisableCopyOnReadDisableCopyOnRead(read_16_disablecopyonread_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp(read_16_disablecopyonread_dense_7_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@@{
Read_17/DisableCopyOnReadDisableCopyOnRead&read_17_disablecopyonread_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp&read_17_disablecopyonread_dense_7_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_18/DisableCopyOnReadDisableCopyOnRead(read_18_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp(read_18_disablecopyonread_dense_8_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:@@{
Read_19/DisableCopyOnReadDisableCopyOnRead&read_19_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp&read_19_disablecopyonread_dense_8_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_dense_9_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@@{
Read_21/DisableCopyOnReadDisableCopyOnRead&read_21_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp&read_21_disablecopyonread_dense_9_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_22/DisableCopyOnReadDisableCopyOnRead*read_22_disablecopyonread_output_nn_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp*read_22_disablecopyonread_output_nn_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:@}
Read_23/DisableCopyOnReadDisableCopyOnRead(read_23_disablecopyonread_output_nn_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp(read_23_disablecopyonread_output_nn_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_24/DisableCopyOnReadDisableCopyOnRead#read_24_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp#read_24_disablecopyonread_iteration^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_25/DisableCopyOnReadDisableCopyOnRead'read_25_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp'read_25_disablecopyonread_learning_rate^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnRead-read_26_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp-read_26_disablecopyonread_adam_m_dense_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_27/DisableCopyOnReadDisableCopyOnRead-read_27_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp-read_27_disablecopyonread_adam_v_dense_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_28/DisableCopyOnReadDisableCopyOnRead+read_28_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp+read_28_disablecopyonread_adam_m_dense_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_29/DisableCopyOnReadDisableCopyOnRead+read_29_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp+read_29_disablecopyonread_adam_v_dense_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_30/DisableCopyOnReadDisableCopyOnRead/read_30_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp/read_30_disablecopyonread_adam_m_dense_1_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_31/DisableCopyOnReadDisableCopyOnRead/read_31_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp/read_31_disablecopyonread_adam_v_dense_1_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_32/DisableCopyOnReadDisableCopyOnRead-read_32_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp-read_32_disablecopyonread_adam_m_dense_1_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_33/DisableCopyOnReadDisableCopyOnRead-read_33_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp-read_33_disablecopyonread_adam_v_dense_1_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_34/DisableCopyOnReadDisableCopyOnRead/read_34_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp/read_34_disablecopyonread_adam_m_dense_2_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_v_dense_2_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_36/DisableCopyOnReadDisableCopyOnRead-read_36_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp-read_36_disablecopyonread_adam_m_dense_2_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_37/DisableCopyOnReadDisableCopyOnRead-read_37_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp-read_37_disablecopyonread_adam_v_dense_2_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_38/DisableCopyOnReadDisableCopyOnRead/read_38_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp/read_38_disablecopyonread_adam_m_dense_3_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_39/DisableCopyOnReadDisableCopyOnRead/read_39_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp/read_39_disablecopyonread_adam_v_dense_3_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_40/DisableCopyOnReadDisableCopyOnRead-read_40_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp-read_40_disablecopyonread_adam_m_dense_3_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_41/DisableCopyOnReadDisableCopyOnRead-read_41_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp-read_41_disablecopyonread_adam_v_dense_3_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_42/DisableCopyOnReadDisableCopyOnRead/read_42_disablecopyonread_adam_m_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp/read_42_disablecopyonread_adam_m_dense_4_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_43/DisableCopyOnReadDisableCopyOnRead/read_43_disablecopyonread_adam_v_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp/read_43_disablecopyonread_adam_v_dense_4_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_44/DisableCopyOnReadDisableCopyOnRead-read_44_disablecopyonread_adam_m_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp-read_44_disablecopyonread_adam_m_dense_4_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_45/DisableCopyOnReadDisableCopyOnRead-read_45_disablecopyonread_adam_v_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp-read_45_disablecopyonread_adam_v_dense_4_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_46/DisableCopyOnReadDisableCopyOnRead3read_46_disablecopyonread_adam_m_output_nn_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp3read_46_disablecopyonread_adam_m_output_nn_kernel_1^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_47/DisableCopyOnReadDisableCopyOnRead3read_47_disablecopyonread_adam_v_output_nn_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp3read_47_disablecopyonread_adam_v_output_nn_kernel_1^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_48/DisableCopyOnReadDisableCopyOnRead1read_48_disablecopyonread_adam_m_output_nn_bias_1"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp1read_48_disablecopyonread_adam_m_output_nn_bias_1^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_49/DisableCopyOnReadDisableCopyOnRead1read_49_disablecopyonread_adam_v_output_nn_bias_1"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp1read_49_disablecopyonread_adam_v_output_nn_bias_1^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_50/DisableCopyOnReadDisableCopyOnRead/read_50_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp/read_50_disablecopyonread_adam_m_dense_5_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0q
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@h
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_51/DisableCopyOnReadDisableCopyOnRead/read_51_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp/read_51_disablecopyonread_adam_v_dense_5_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0q
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@h
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_52/DisableCopyOnReadDisableCopyOnRead-read_52_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp-read_52_disablecopyonread_adam_m_dense_5_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_53/DisableCopyOnReadDisableCopyOnRead-read_53_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp-read_53_disablecopyonread_adam_v_dense_5_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_54/DisableCopyOnReadDisableCopyOnRead/read_54_disablecopyonread_adam_m_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp/read_54_disablecopyonread_adam_m_dense_6_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_55/DisableCopyOnReadDisableCopyOnRead/read_55_disablecopyonread_adam_v_dense_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp/read_55_disablecopyonread_adam_v_dense_6_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_56/DisableCopyOnReadDisableCopyOnRead-read_56_disablecopyonread_adam_m_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp-read_56_disablecopyonread_adam_m_dense_6_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_57/DisableCopyOnReadDisableCopyOnRead-read_57_disablecopyonread_adam_v_dense_6_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp-read_57_disablecopyonread_adam_v_dense_6_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_58/DisableCopyOnReadDisableCopyOnRead/read_58_disablecopyonread_adam_m_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp/read_58_disablecopyonread_adam_m_dense_7_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_59/DisableCopyOnReadDisableCopyOnRead/read_59_disablecopyonread_adam_v_dense_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp/read_59_disablecopyonread_adam_v_dense_7_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_60/DisableCopyOnReadDisableCopyOnRead-read_60_disablecopyonread_adam_m_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp-read_60_disablecopyonread_adam_m_dense_7_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_61/DisableCopyOnReadDisableCopyOnRead-read_61_disablecopyonread_adam_v_dense_7_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp-read_61_disablecopyonread_adam_v_dense_7_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_62/DisableCopyOnReadDisableCopyOnRead/read_62_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp/read_62_disablecopyonread_adam_m_dense_8_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_63/DisableCopyOnReadDisableCopyOnRead/read_63_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp/read_63_disablecopyonread_adam_v_dense_8_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_64/DisableCopyOnReadDisableCopyOnRead-read_64_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp-read_64_disablecopyonread_adam_m_dense_8_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_65/DisableCopyOnReadDisableCopyOnRead-read_65_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp-read_65_disablecopyonread_adam_v_dense_8_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_66/DisableCopyOnReadDisableCopyOnRead/read_66_disablecopyonread_adam_m_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp/read_66_disablecopyonread_adam_m_dense_9_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_67/DisableCopyOnReadDisableCopyOnRead/read_67_disablecopyonread_adam_v_dense_9_kernel"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp/read_67_disablecopyonread_adam_v_dense_9_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_68/DisableCopyOnReadDisableCopyOnRead-read_68_disablecopyonread_adam_m_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp-read_68_disablecopyonread_adam_m_dense_9_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_69/DisableCopyOnReadDisableCopyOnRead-read_69_disablecopyonread_adam_v_dense_9_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp-read_69_disablecopyonread_adam_v_dense_9_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_70/DisableCopyOnReadDisableCopyOnRead1read_70_disablecopyonread_adam_m_output_nn_kernel"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp1read_70_disablecopyonread_adam_m_output_nn_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_71/DisableCopyOnReadDisableCopyOnRead1read_71_disablecopyonread_adam_v_output_nn_kernel"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp1read_71_disablecopyonread_adam_v_output_nn_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0p
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@g
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_72/DisableCopyOnReadDisableCopyOnRead/read_72_disablecopyonread_adam_m_output_nn_bias"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp/read_72_disablecopyonread_adam_m_output_nn_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_73/DisableCopyOnReadDisableCopyOnRead/read_73_disablecopyonread_adam_v_output_nn_bias"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp/read_73_disablecopyonread_adam_v_output_nn_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:t
Read_74/DisableCopyOnReadDisableCopyOnReadread_74_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOpread_74_disablecopyonread_total^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_75/DisableCopyOnReadDisableCopyOnReadread_75_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOpread_75_disablecopyonread_count^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*�
value�B�MB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*�
value�B�MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *[
dtypesQ
O2M	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_152Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_153IdentityIdentity_152:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_153Identity_153:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:M

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_output_NN_layer_call_fn_779745

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_777761o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
)__inference_Group_NN_layer_call_fn_779356

inputs
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_777711

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778286

inputs!
dense_5_778255:	�@
dense_5_778257:@ 
dense_6_778260:@@
dense_6_778262:@ 
dense_7_778265:@@
dense_7_778267:@ 
dense_8_778270:@@
dense_8_778272:@ 
dense_9_778275:@@
dense_9_778277:@"
output_nn_778280:@
output_nn_778282:
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_778255dense_5_778257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_778061�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_778260dense_6_778262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_778078�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_778265dense_7_778267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_778095�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_778270dense_8_778272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_778112�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_778275dense_9_778277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_778129�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0output_nn_778280output_nn_778282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_778145y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777839

inputs
dense_777808:	� 
dense_777810:  
dense_1_777813:  
dense_1_777815:  
dense_2_777818:  
dense_2_777820:  
dense_3_777823:  
dense_3_777825:  
dense_4_777828:  
dense_4_777830: "
output_nn_777833: 
output_nn_777835:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_777808dense_777810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_777677�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_777813dense_1_777815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_777694�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_777818dense_2_777820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_777711�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_777823dense_3_777825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_777728�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_777828dense_4_777830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_777745�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0output_nn_777833output_nn_777835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_777761y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_778129

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_778112

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�2
�	
D__inference_Group_NN_layer_call_and_return_conditional_losses_779429

inputs7
$dense_matmul_readvariableop_resource:	� 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource:  5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: :
(output_nn_matmul_readvariableop_resource: 7
)output_nn_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output_NN/MatMulMatMuldense_4/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 output_NN/BiasAdd/ReadVariableOpReadVariableOp)output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_NN/BiasAddBiasAddoutput_NN/MatMul:product:0(output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentityoutput_NN/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777802
dense_input
dense_777771:	� 
dense_777773:  
dense_1_777776:  
dense_1_777778:  
dense_2_777781:  
dense_2_777783:  
dense_3_777786:  
dense_3_777788:  
dense_4_777791:  
dense_4_777793: "
output_nn_777796: 
output_nn_777798:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_777771dense_777773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_777677�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_777776dense_1_777778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_777694�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_777781dense_2_777783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_777711�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_777786dense_3_777788*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_777728�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_777791dense_4_777793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_777745�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0output_nn_777796output_nn_777798*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_777761y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
(__inference_dense_9_layer_call_fn_779843

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_778129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
A__inference_dense_layer_call_and_return_conditional_losses_777677

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_777663
input_group
input_techniqueG
4model1_group_nn_dense_matmul_readvariableop_resource:	� C
5model1_group_nn_dense_biasadd_readvariableop_resource: H
6model1_group_nn_dense_1_matmul_readvariableop_resource:  E
7model1_group_nn_dense_1_biasadd_readvariableop_resource: H
6model1_group_nn_dense_2_matmul_readvariableop_resource:  E
7model1_group_nn_dense_2_biasadd_readvariableop_resource: H
6model1_group_nn_dense_3_matmul_readvariableop_resource:  E
7model1_group_nn_dense_3_biasadd_readvariableop_resource: H
6model1_group_nn_dense_4_matmul_readvariableop_resource:  E
7model1_group_nn_dense_4_biasadd_readvariableop_resource: J
8model1_group_nn_output_nn_matmul_readvariableop_resource: G
9model1_group_nn_output_nn_biasadd_readvariableop_resource:M
:model1_technique_nn_dense_5_matmul_readvariableop_resource:	�@I
;model1_technique_nn_dense_5_biasadd_readvariableop_resource:@L
:model1_technique_nn_dense_6_matmul_readvariableop_resource:@@I
;model1_technique_nn_dense_6_biasadd_readvariableop_resource:@L
:model1_technique_nn_dense_7_matmul_readvariableop_resource:@@I
;model1_technique_nn_dense_7_biasadd_readvariableop_resource:@L
:model1_technique_nn_dense_8_matmul_readvariableop_resource:@@I
;model1_technique_nn_dense_8_biasadd_readvariableop_resource:@L
:model1_technique_nn_dense_9_matmul_readvariableop_resource:@@I
;model1_technique_nn_dense_9_biasadd_readvariableop_resource:@N
<model1_technique_nn_output_nn_matmul_readvariableop_resource:@K
=model1_technique_nn_output_nn_biasadd_readvariableop_resource:
identity��,model1/Group_NN/dense/BiasAdd/ReadVariableOp�+model1/Group_NN/dense/MatMul/ReadVariableOp�.model1/Group_NN/dense_1/BiasAdd/ReadVariableOp�-model1/Group_NN/dense_1/MatMul/ReadVariableOp�.model1/Group_NN/dense_2/BiasAdd/ReadVariableOp�-model1/Group_NN/dense_2/MatMul/ReadVariableOp�.model1/Group_NN/dense_3/BiasAdd/ReadVariableOp�-model1/Group_NN/dense_3/MatMul/ReadVariableOp�.model1/Group_NN/dense_4/BiasAdd/ReadVariableOp�-model1/Group_NN/dense_4/MatMul/ReadVariableOp�0model1/Group_NN/output_NN/BiasAdd/ReadVariableOp�/model1/Group_NN/output_NN/MatMul/ReadVariableOp�2model1/Technique_NN/dense_5/BiasAdd/ReadVariableOp�1model1/Technique_NN/dense_5/MatMul/ReadVariableOp�2model1/Technique_NN/dense_6/BiasAdd/ReadVariableOp�1model1/Technique_NN/dense_6/MatMul/ReadVariableOp�2model1/Technique_NN/dense_7/BiasAdd/ReadVariableOp�1model1/Technique_NN/dense_7/MatMul/ReadVariableOp�2model1/Technique_NN/dense_8/BiasAdd/ReadVariableOp�1model1/Technique_NN/dense_8/MatMul/ReadVariableOp�2model1/Technique_NN/dense_9/BiasAdd/ReadVariableOp�1model1/Technique_NN/dense_9/MatMul/ReadVariableOp�4model1/Technique_NN/output_NN/BiasAdd/ReadVariableOp�3model1/Technique_NN/output_NN/MatMul/ReadVariableOp�
+model1/Group_NN/dense/MatMul/ReadVariableOpReadVariableOp4model1_group_nn_dense_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
model1/Group_NN/dense/MatMulMatMulinput_group3model1/Group_NN/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
,model1/Group_NN/dense/BiasAdd/ReadVariableOpReadVariableOp5model1_group_nn_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model1/Group_NN/dense/BiasAddBiasAdd&model1/Group_NN/dense/MatMul:product:04model1/Group_NN/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
-model1/Group_NN/dense_1/MatMul/ReadVariableOpReadVariableOp6model1_group_nn_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model1/Group_NN/dense_1/MatMulMatMul&model1/Group_NN/dense/BiasAdd:output:05model1/Group_NN/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.model1/Group_NN/dense_1/BiasAdd/ReadVariableOpReadVariableOp7model1_group_nn_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model1/Group_NN/dense_1/BiasAddBiasAdd(model1/Group_NN/dense_1/MatMul:product:06model1/Group_NN/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1/Group_NN/dense_1/ReluRelu(model1/Group_NN/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
-model1/Group_NN/dense_2/MatMul/ReadVariableOpReadVariableOp6model1_group_nn_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model1/Group_NN/dense_2/MatMulMatMul*model1/Group_NN/dense_1/Relu:activations:05model1/Group_NN/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.model1/Group_NN/dense_2/BiasAdd/ReadVariableOpReadVariableOp7model1_group_nn_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model1/Group_NN/dense_2/BiasAddBiasAdd(model1/Group_NN/dense_2/MatMul:product:06model1/Group_NN/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1/Group_NN/dense_2/ReluRelu(model1/Group_NN/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
-model1/Group_NN/dense_3/MatMul/ReadVariableOpReadVariableOp6model1_group_nn_dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model1/Group_NN/dense_3/MatMulMatMul*model1/Group_NN/dense_2/Relu:activations:05model1/Group_NN/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.model1/Group_NN/dense_3/BiasAdd/ReadVariableOpReadVariableOp7model1_group_nn_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model1/Group_NN/dense_3/BiasAddBiasAdd(model1/Group_NN/dense_3/MatMul:product:06model1/Group_NN/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1/Group_NN/dense_3/ReluRelu(model1/Group_NN/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
-model1/Group_NN/dense_4/MatMul/ReadVariableOpReadVariableOp6model1_group_nn_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
model1/Group_NN/dense_4/MatMulMatMul*model1/Group_NN/dense_3/Relu:activations:05model1/Group_NN/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
.model1/Group_NN/dense_4/BiasAdd/ReadVariableOpReadVariableOp7model1_group_nn_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model1/Group_NN/dense_4/BiasAddBiasAdd(model1/Group_NN/dense_4/MatMul:product:06model1/Group_NN/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1/Group_NN/dense_4/ReluRelu(model1/Group_NN/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
/model1/Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp8model1_group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 model1/Group_NN/output_NN/MatMulMatMul*model1/Group_NN/dense_4/Relu:activations:07model1/Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0model1/Group_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp9model1_group_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!model1/Group_NN/output_NN/BiasAddBiasAdd*model1/Group_NN/output_NN/MatMul:product:08model1/Group_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1model1/Technique_NN/dense_5/MatMul/ReadVariableOpReadVariableOp:model1_technique_nn_dense_5_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
"model1/Technique_NN/dense_5/MatMulMatMulinput_technique9model1/Technique_NN/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2model1/Technique_NN/dense_5/BiasAdd/ReadVariableOpReadVariableOp;model1_technique_nn_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#model1/Technique_NN/dense_5/BiasAddBiasAdd,model1/Technique_NN/dense_5/MatMul:product:0:model1/Technique_NN/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1model1/Technique_NN/dense_6/MatMul/ReadVariableOpReadVariableOp:model1_technique_nn_dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"model1/Technique_NN/dense_6/MatMulMatMul,model1/Technique_NN/dense_5/BiasAdd:output:09model1/Technique_NN/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2model1/Technique_NN/dense_6/BiasAdd/ReadVariableOpReadVariableOp;model1_technique_nn_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#model1/Technique_NN/dense_6/BiasAddBiasAdd,model1/Technique_NN/dense_6/MatMul:product:0:model1/Technique_NN/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 model1/Technique_NN/dense_6/ReluRelu,model1/Technique_NN/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1model1/Technique_NN/dense_7/MatMul/ReadVariableOpReadVariableOp:model1_technique_nn_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"model1/Technique_NN/dense_7/MatMulMatMul.model1/Technique_NN/dense_6/Relu:activations:09model1/Technique_NN/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2model1/Technique_NN/dense_7/BiasAdd/ReadVariableOpReadVariableOp;model1_technique_nn_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#model1/Technique_NN/dense_7/BiasAddBiasAdd,model1/Technique_NN/dense_7/MatMul:product:0:model1/Technique_NN/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 model1/Technique_NN/dense_7/ReluRelu,model1/Technique_NN/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1model1/Technique_NN/dense_8/MatMul/ReadVariableOpReadVariableOp:model1_technique_nn_dense_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"model1/Technique_NN/dense_8/MatMulMatMul.model1/Technique_NN/dense_7/Relu:activations:09model1/Technique_NN/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2model1/Technique_NN/dense_8/BiasAdd/ReadVariableOpReadVariableOp;model1_technique_nn_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#model1/Technique_NN/dense_8/BiasAddBiasAdd,model1/Technique_NN/dense_8/MatMul:product:0:model1/Technique_NN/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 model1/Technique_NN/dense_8/ReluRelu,model1/Technique_NN/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
1model1/Technique_NN/dense_9/MatMul/ReadVariableOpReadVariableOp:model1_technique_nn_dense_9_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
"model1/Technique_NN/dense_9/MatMulMatMul.model1/Technique_NN/dense_8/Relu:activations:09model1/Technique_NN/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
2model1/Technique_NN/dense_9/BiasAdd/ReadVariableOpReadVariableOp;model1_technique_nn_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
#model1/Technique_NN/dense_9/BiasAddBiasAdd,model1/Technique_NN/dense_9/MatMul:product:0:model1/Technique_NN/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 model1/Technique_NN/dense_9/ReluRelu,model1/Technique_NN/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
3model1/Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp<model1_technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
$model1/Technique_NN/output_NN/MatMulMatMul.model1/Technique_NN/dense_9/Relu:activations:0;model1/Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4model1/Technique_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp=model1_technique_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
%model1/Technique_NN/output_NN/BiasAddBiasAdd.model1/Technique_NN/output_NN/MatMul:product:0<model1/Technique_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model1/l2_normalize/SquareSquare*model1/Group_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������k
)model1/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model1/l2_normalize/SumSummodel1/l2_normalize/Square:y:02model1/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(b
model1/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
model1/l2_normalize/MaximumMaximum model1/l2_normalize/Sum:output:0&model1/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������u
model1/l2_normalize/RsqrtRsqrtmodel1/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
model1/l2_normalizeMul*model1/Group_NN/output_NN/BiasAdd:output:0model1/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
model1/l2_normalize_1/SquareSquare.model1/Technique_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
+model1/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model1/l2_normalize_1/SumSum model1/l2_normalize_1/Square:y:04model1/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(d
model1/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
model1/l2_normalize_1/MaximumMaximum"model1/l2_normalize_1/Sum:output:0(model1/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������y
model1/l2_normalize_1/RsqrtRsqrt!model1/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
model1/l2_normalize_1Mul.model1/Technique_NN/output_NN/BiasAdd:output:0model1/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������[
model1/dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model1/dot/ExpandDims
ExpandDimsmodel1/l2_normalize:z:0"model1/dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������]
model1/dot/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model1/dot/ExpandDims_1
ExpandDimsmodel1/l2_normalize_1:z:0$model1/dot/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
model1/dot/MatMulBatchMatMulV2model1/dot/ExpandDims:output:0 model1/dot/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������h
model1/dot/ShapeShapemodel1/dot/MatMul:output:0*
T0*
_output_shapes
::���
model1/dot/SqueezeSqueezemodel1/dot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
j
IdentityIdentitymodel1/dot/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp-^model1/Group_NN/dense/BiasAdd/ReadVariableOp,^model1/Group_NN/dense/MatMul/ReadVariableOp/^model1/Group_NN/dense_1/BiasAdd/ReadVariableOp.^model1/Group_NN/dense_1/MatMul/ReadVariableOp/^model1/Group_NN/dense_2/BiasAdd/ReadVariableOp.^model1/Group_NN/dense_2/MatMul/ReadVariableOp/^model1/Group_NN/dense_3/BiasAdd/ReadVariableOp.^model1/Group_NN/dense_3/MatMul/ReadVariableOp/^model1/Group_NN/dense_4/BiasAdd/ReadVariableOp.^model1/Group_NN/dense_4/MatMul/ReadVariableOp1^model1/Group_NN/output_NN/BiasAdd/ReadVariableOp0^model1/Group_NN/output_NN/MatMul/ReadVariableOp3^model1/Technique_NN/dense_5/BiasAdd/ReadVariableOp2^model1/Technique_NN/dense_5/MatMul/ReadVariableOp3^model1/Technique_NN/dense_6/BiasAdd/ReadVariableOp2^model1/Technique_NN/dense_6/MatMul/ReadVariableOp3^model1/Technique_NN/dense_7/BiasAdd/ReadVariableOp2^model1/Technique_NN/dense_7/MatMul/ReadVariableOp3^model1/Technique_NN/dense_8/BiasAdd/ReadVariableOp2^model1/Technique_NN/dense_8/MatMul/ReadVariableOp3^model1/Technique_NN/dense_9/BiasAdd/ReadVariableOp2^model1/Technique_NN/dense_9/MatMul/ReadVariableOp5^model1/Technique_NN/output_NN/BiasAdd/ReadVariableOp4^model1/Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2\
,model1/Group_NN/dense/BiasAdd/ReadVariableOp,model1/Group_NN/dense/BiasAdd/ReadVariableOp2Z
+model1/Group_NN/dense/MatMul/ReadVariableOp+model1/Group_NN/dense/MatMul/ReadVariableOp2`
.model1/Group_NN/dense_1/BiasAdd/ReadVariableOp.model1/Group_NN/dense_1/BiasAdd/ReadVariableOp2^
-model1/Group_NN/dense_1/MatMul/ReadVariableOp-model1/Group_NN/dense_1/MatMul/ReadVariableOp2`
.model1/Group_NN/dense_2/BiasAdd/ReadVariableOp.model1/Group_NN/dense_2/BiasAdd/ReadVariableOp2^
-model1/Group_NN/dense_2/MatMul/ReadVariableOp-model1/Group_NN/dense_2/MatMul/ReadVariableOp2`
.model1/Group_NN/dense_3/BiasAdd/ReadVariableOp.model1/Group_NN/dense_3/BiasAdd/ReadVariableOp2^
-model1/Group_NN/dense_3/MatMul/ReadVariableOp-model1/Group_NN/dense_3/MatMul/ReadVariableOp2`
.model1/Group_NN/dense_4/BiasAdd/ReadVariableOp.model1/Group_NN/dense_4/BiasAdd/ReadVariableOp2^
-model1/Group_NN/dense_4/MatMul/ReadVariableOp-model1/Group_NN/dense_4/MatMul/ReadVariableOp2d
0model1/Group_NN/output_NN/BiasAdd/ReadVariableOp0model1/Group_NN/output_NN/BiasAdd/ReadVariableOp2b
/model1/Group_NN/output_NN/MatMul/ReadVariableOp/model1/Group_NN/output_NN/MatMul/ReadVariableOp2h
2model1/Technique_NN/dense_5/BiasAdd/ReadVariableOp2model1/Technique_NN/dense_5/BiasAdd/ReadVariableOp2f
1model1/Technique_NN/dense_5/MatMul/ReadVariableOp1model1/Technique_NN/dense_5/MatMul/ReadVariableOp2h
2model1/Technique_NN/dense_6/BiasAdd/ReadVariableOp2model1/Technique_NN/dense_6/BiasAdd/ReadVariableOp2f
1model1/Technique_NN/dense_6/MatMul/ReadVariableOp1model1/Technique_NN/dense_6/MatMul/ReadVariableOp2h
2model1/Technique_NN/dense_7/BiasAdd/ReadVariableOp2model1/Technique_NN/dense_7/BiasAdd/ReadVariableOp2f
1model1/Technique_NN/dense_7/MatMul/ReadVariableOp1model1/Technique_NN/dense_7/MatMul/ReadVariableOp2h
2model1/Technique_NN/dense_8/BiasAdd/ReadVariableOp2model1/Technique_NN/dense_8/BiasAdd/ReadVariableOp2f
1model1/Technique_NN/dense_8/MatMul/ReadVariableOp1model1/Technique_NN/dense_8/MatMul/ReadVariableOp2h
2model1/Technique_NN/dense_9/BiasAdd/ReadVariableOp2model1/Technique_NN/dense_9/BiasAdd/ReadVariableOp2f
1model1/Technique_NN/dense_9/MatMul/ReadVariableOp1model1/Technique_NN/dense_9/MatMul/ReadVariableOp2l
4model1/Technique_NN/output_NN/BiasAdd/ReadVariableOp4model1/Technique_NN/output_NN/BiasAdd/ReadVariableOp2j
3model1/Technique_NN/output_NN/MatMul/ReadVariableOp3model1/Technique_NN/output_NN/MatMul/ReadVariableOp:YU
(
_output_shapes
:����������
)
_user_specified_nameinput_Technique:U Q
(
_output_shapes
:����������
%
_user_specified_nameinput_Group
�
�
(__inference_dense_1_layer_call_fn_779665

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_777694o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
i
?__inference_dot_layer_call_and_return_conditional_losses_778512

inputs
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :o

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:���������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :u
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
)__inference_Group_NN_layer_call_fn_777929
dense_input
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777902o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�	
�
C__inference_dense_5_layer_call_and_return_conditional_losses_779774

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_779736

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778223

inputs!
dense_5_778192:	�@
dense_5_778194:@ 
dense_6_778197:@@
dense_6_778199:@ 
dense_7_778202:@@
dense_7_778204:@ 
dense_8_778207:@@
dense_8_778209:@ 
dense_9_778212:@@
dense_9_778214:@"
output_nn_778217:@
output_nn_778219:
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_778192dense_5_778194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_778061�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_778197dense_6_778199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_778078�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_778202dense_7_778204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_778095�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_778207dense_8_778209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_778112�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_778212dense_9_778214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_778129�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0output_nn_778217output_nn_778219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_778145y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ƌ
�
B__inference_model1_layer_call_and_return_conditional_losses_779327
inputs_input_group
inputs_input_technique@
-group_nn_dense_matmul_readvariableop_resource:	� <
.group_nn_dense_biasadd_readvariableop_resource: A
/group_nn_dense_1_matmul_readvariableop_resource:  >
0group_nn_dense_1_biasadd_readvariableop_resource: A
/group_nn_dense_2_matmul_readvariableop_resource:  >
0group_nn_dense_2_biasadd_readvariableop_resource: A
/group_nn_dense_3_matmul_readvariableop_resource:  >
0group_nn_dense_3_biasadd_readvariableop_resource: A
/group_nn_dense_4_matmul_readvariableop_resource:  >
0group_nn_dense_4_biasadd_readvariableop_resource: C
1group_nn_output_nn_matmul_readvariableop_resource: @
2group_nn_output_nn_biasadd_readvariableop_resource:F
3technique_nn_dense_5_matmul_readvariableop_resource:	�@B
4technique_nn_dense_5_biasadd_readvariableop_resource:@E
3technique_nn_dense_6_matmul_readvariableop_resource:@@B
4technique_nn_dense_6_biasadd_readvariableop_resource:@E
3technique_nn_dense_7_matmul_readvariableop_resource:@@B
4technique_nn_dense_7_biasadd_readvariableop_resource:@E
3technique_nn_dense_8_matmul_readvariableop_resource:@@B
4technique_nn_dense_8_biasadd_readvariableop_resource:@E
3technique_nn_dense_9_matmul_readvariableop_resource:@@B
4technique_nn_dense_9_biasadd_readvariableop_resource:@G
5technique_nn_output_nn_matmul_readvariableop_resource:@D
6technique_nn_output_nn_biasadd_readvariableop_resource:
identity��%Group_NN/dense/BiasAdd/ReadVariableOp�$Group_NN/dense/MatMul/ReadVariableOp�'Group_NN/dense_1/BiasAdd/ReadVariableOp�&Group_NN/dense_1/MatMul/ReadVariableOp�'Group_NN/dense_2/BiasAdd/ReadVariableOp�&Group_NN/dense_2/MatMul/ReadVariableOp�'Group_NN/dense_3/BiasAdd/ReadVariableOp�&Group_NN/dense_3/MatMul/ReadVariableOp�'Group_NN/dense_4/BiasAdd/ReadVariableOp�&Group_NN/dense_4/MatMul/ReadVariableOp�)Group_NN/output_NN/BiasAdd/ReadVariableOp�(Group_NN/output_NN/MatMul/ReadVariableOp�+Technique_NN/dense_5/BiasAdd/ReadVariableOp�*Technique_NN/dense_5/MatMul/ReadVariableOp�+Technique_NN/dense_6/BiasAdd/ReadVariableOp�*Technique_NN/dense_6/MatMul/ReadVariableOp�+Technique_NN/dense_7/BiasAdd/ReadVariableOp�*Technique_NN/dense_7/MatMul/ReadVariableOp�+Technique_NN/dense_8/BiasAdd/ReadVariableOp�*Technique_NN/dense_8/MatMul/ReadVariableOp�+Technique_NN/dense_9/BiasAdd/ReadVariableOp�*Technique_NN/dense_9/MatMul/ReadVariableOp�-Technique_NN/output_NN/BiasAdd/ReadVariableOp�,Technique_NN/output_NN/MatMul/ReadVariableOp�
$Group_NN/dense/MatMul/ReadVariableOpReadVariableOp-group_nn_dense_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
Group_NN/dense/MatMulMatMulinputs_input_group,Group_NN/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%Group_NN/dense/BiasAdd/ReadVariableOpReadVariableOp.group_nn_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense/BiasAddBiasAddGroup_NN/dense/MatMul:product:0-Group_NN/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_1/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_1/MatMulMatMulGroup_NN/dense/BiasAdd:output:0.Group_NN/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_1/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_1/BiasAddBiasAdd!Group_NN/dense_1/MatMul:product:0/Group_NN/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_1/ReluRelu!Group_NN/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_2/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_2/MatMulMatMul#Group_NN/dense_1/Relu:activations:0.Group_NN/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_2/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_2/BiasAddBiasAdd!Group_NN/dense_2/MatMul:product:0/Group_NN/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_2/ReluRelu!Group_NN/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_3/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_3/MatMulMatMul#Group_NN/dense_2/Relu:activations:0.Group_NN/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_3/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_3/BiasAddBiasAdd!Group_NN/dense_3/MatMul:product:0/Group_NN/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_3/ReluRelu!Group_NN/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_4/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_4/MatMulMatMul#Group_NN/dense_3/Relu:activations:0.Group_NN/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_4/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_4/BiasAddBiasAdd!Group_NN/dense_4/MatMul:product:0/Group_NN/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_4/ReluRelu!Group_NN/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp1group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Group_NN/output_NN/MatMulMatMul#Group_NN/dense_4/Relu:activations:00Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)Group_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp2group_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Group_NN/output_NN/BiasAddBiasAdd#Group_NN/output_NN/MatMul:product:01Group_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*Technique_NN/dense_5/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_5_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Technique_NN/dense_5/MatMulMatMulinputs_input_technique2Technique_NN/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_5/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_5/BiasAddBiasAdd%Technique_NN/dense_5/MatMul:product:03Technique_NN/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_6/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_6/MatMulMatMul%Technique_NN/dense_5/BiasAdd:output:02Technique_NN/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_6/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_6/BiasAddBiasAdd%Technique_NN/dense_6/MatMul:product:03Technique_NN/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_6/ReluRelu%Technique_NN/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_7/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_7/MatMulMatMul'Technique_NN/dense_6/Relu:activations:02Technique_NN/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_7/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_7/BiasAddBiasAdd%Technique_NN/dense_7/MatMul:product:03Technique_NN/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_7/ReluRelu%Technique_NN/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_8/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_8/MatMulMatMul'Technique_NN/dense_7/Relu:activations:02Technique_NN/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_8/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_8/BiasAddBiasAdd%Technique_NN/dense_8/MatMul:product:03Technique_NN/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_8/ReluRelu%Technique_NN/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_9/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_9_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_9/MatMulMatMul'Technique_NN/dense_8/Relu:activations:02Technique_NN/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_9/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_9/BiasAddBiasAdd%Technique_NN/dense_9/MatMul:product:03Technique_NN/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_9/ReluRelu%Technique_NN/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp5technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
Technique_NN/output_NN/MatMulMatMul'Technique_NN/dense_9/Relu:activations:04Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-Technique_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp6technique_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Technique_NN/output_NN/BiasAddBiasAdd'Technique_NN/output_NN/MatMul:product:05Technique_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
l2_normalize/SquareSquare#Group_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalizeMul#Group_NN/output_NN/BiasAdd:output:0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������z
l2_normalize_1/SquareSquare'Technique_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalize_1Mul'Technique_NN/output_NN/BiasAdd:output:0l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������T
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims
ExpandDimsl2_normalize:z:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������V
dot/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims_1
ExpandDimsl2_normalize_1:z:0dot/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������

dot/MatMulBatchMatMulV2dot/ExpandDims:output:0dot/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������Z
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
::��t
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
c
IdentityIdentitydot/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^Group_NN/dense/BiasAdd/ReadVariableOp%^Group_NN/dense/MatMul/ReadVariableOp(^Group_NN/dense_1/BiasAdd/ReadVariableOp'^Group_NN/dense_1/MatMul/ReadVariableOp(^Group_NN/dense_2/BiasAdd/ReadVariableOp'^Group_NN/dense_2/MatMul/ReadVariableOp(^Group_NN/dense_3/BiasAdd/ReadVariableOp'^Group_NN/dense_3/MatMul/ReadVariableOp(^Group_NN/dense_4/BiasAdd/ReadVariableOp'^Group_NN/dense_4/MatMul/ReadVariableOp*^Group_NN/output_NN/BiasAdd/ReadVariableOp)^Group_NN/output_NN/MatMul/ReadVariableOp,^Technique_NN/dense_5/BiasAdd/ReadVariableOp+^Technique_NN/dense_5/MatMul/ReadVariableOp,^Technique_NN/dense_6/BiasAdd/ReadVariableOp+^Technique_NN/dense_6/MatMul/ReadVariableOp,^Technique_NN/dense_7/BiasAdd/ReadVariableOp+^Technique_NN/dense_7/MatMul/ReadVariableOp,^Technique_NN/dense_8/BiasAdd/ReadVariableOp+^Technique_NN/dense_8/MatMul/ReadVariableOp,^Technique_NN/dense_9/BiasAdd/ReadVariableOp+^Technique_NN/dense_9/MatMul/ReadVariableOp.^Technique_NN/output_NN/BiasAdd/ReadVariableOp-^Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2N
%Group_NN/dense/BiasAdd/ReadVariableOp%Group_NN/dense/BiasAdd/ReadVariableOp2L
$Group_NN/dense/MatMul/ReadVariableOp$Group_NN/dense/MatMul/ReadVariableOp2R
'Group_NN/dense_1/BiasAdd/ReadVariableOp'Group_NN/dense_1/BiasAdd/ReadVariableOp2P
&Group_NN/dense_1/MatMul/ReadVariableOp&Group_NN/dense_1/MatMul/ReadVariableOp2R
'Group_NN/dense_2/BiasAdd/ReadVariableOp'Group_NN/dense_2/BiasAdd/ReadVariableOp2P
&Group_NN/dense_2/MatMul/ReadVariableOp&Group_NN/dense_2/MatMul/ReadVariableOp2R
'Group_NN/dense_3/BiasAdd/ReadVariableOp'Group_NN/dense_3/BiasAdd/ReadVariableOp2P
&Group_NN/dense_3/MatMul/ReadVariableOp&Group_NN/dense_3/MatMul/ReadVariableOp2R
'Group_NN/dense_4/BiasAdd/ReadVariableOp'Group_NN/dense_4/BiasAdd/ReadVariableOp2P
&Group_NN/dense_4/MatMul/ReadVariableOp&Group_NN/dense_4/MatMul/ReadVariableOp2V
)Group_NN/output_NN/BiasAdd/ReadVariableOp)Group_NN/output_NN/BiasAdd/ReadVariableOp2T
(Group_NN/output_NN/MatMul/ReadVariableOp(Group_NN/output_NN/MatMul/ReadVariableOp2Z
+Technique_NN/dense_5/BiasAdd/ReadVariableOp+Technique_NN/dense_5/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_5/MatMul/ReadVariableOp*Technique_NN/dense_5/MatMul/ReadVariableOp2Z
+Technique_NN/dense_6/BiasAdd/ReadVariableOp+Technique_NN/dense_6/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_6/MatMul/ReadVariableOp*Technique_NN/dense_6/MatMul/ReadVariableOp2Z
+Technique_NN/dense_7/BiasAdd/ReadVariableOp+Technique_NN/dense_7/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_7/MatMul/ReadVariableOp*Technique_NN/dense_7/MatMul/ReadVariableOp2Z
+Technique_NN/dense_8/BiasAdd/ReadVariableOp+Technique_NN/dense_8/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_8/MatMul/ReadVariableOp*Technique_NN/dense_8/MatMul/ReadVariableOp2Z
+Technique_NN/dense_9/BiasAdd/ReadVariableOp+Technique_NN/dense_9/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_9/MatMul/ReadVariableOp*Technique_NN/dense_9/MatMul/ReadVariableOp2^
-Technique_NN/output_NN/BiasAdd/ReadVariableOp-Technique_NN/output_NN/BiasAdd/ReadVariableOp2\
,Technique_NN/output_NN/MatMul/ReadVariableOp,Technique_NN/output_NN/MatMul/ReadVariableOp:`\
(
_output_shapes
:����������
0
_user_specified_nameinputs_input_technique:\ X
(
_output_shapes
:����������
,
_user_specified_nameinputs_input_group
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_777745

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
� 
�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778152
dense_5_input!
dense_5_778062:	�@
dense_5_778064:@ 
dense_6_778079:@@
dense_6_778081:@ 
dense_7_778096:@@
dense_7_778098:@ 
dense_8_778113:@@
dense_8_778115:@ 
dense_9_778130:@@
dense_9_778132:@"
output_nn_778146:@
output_nn_778148:
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCalldense_5_inputdense_5_778062dense_5_778064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_778061�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_778079dense_6_778081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_778078�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_778096dense_7_778098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_778095�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_778113dense_8_778115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_778112�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_778130dense_9_778132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_778129�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0output_nn_778146output_nn_778148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_778145y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_namedense_5_input
�

�
)__inference_Group_NN_layer_call_fn_779385

inputs
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777902o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model1_layer_call_fn_779115
inputs_input_group
inputs_input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_input_groupinputs_input_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model1_layer_call_and_return_conditional_losses_778783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:`\
(
_output_shapes
:����������
0
_user_specified_nameinputs_input_technique:\ X
(
_output_shapes
:����������
,
_user_specified_nameinputs_input_group
�
�
$__inference_signature_wrapper_779007
input_group
input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_groupinput_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_777663o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:YU
(
_output_shapes
:����������
)
_user_specified_nameinput_Technique:U Q
(
_output_shapes
:����������
%
_user_specified_nameinput_Group
�'
�
B__inference_model1_layer_call_and_return_conditional_losses_778585
input_group
input_technique"
group_nn_778519:	� 
group_nn_778521: !
group_nn_778523:  
group_nn_778525: !
group_nn_778527:  
group_nn_778529: !
group_nn_778531:  
group_nn_778533: !
group_nn_778535:  
group_nn_778537: !
group_nn_778539: 
group_nn_778541:&
technique_nn_778544:	�@!
technique_nn_778546:@%
technique_nn_778548:@@!
technique_nn_778550:@%
technique_nn_778552:@@!
technique_nn_778554:@%
technique_nn_778556:@@!
technique_nn_778558:@%
technique_nn_778560:@@!
technique_nn_778562:@%
technique_nn_778564:@!
technique_nn_778566:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinput_groupgroup_nn_778519group_nn_778521group_nn_778523group_nn_778525group_nn_778527group_nn_778529group_nn_778531group_nn_778533group_nn_778535group_nn_778537group_nn_778539group_nn_778541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777902�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinput_techniquetechnique_nn_778544technique_nn_778546technique_nn_778548technique_nn_778550technique_nn_778552technique_nn_778554technique_nn_778556technique_nn_778558technique_nn_778560technique_nn_778562technique_nn_778564technique_nn_778566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778286z
l2_normalize/SquareSquare)Group_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalizeMul)Group_NN/StatefulPartitionedCall:output:0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
l2_normalize_1/SquareSquare-Technique_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalize_1Mul-Technique_NN/StatefulPartitionedCall:output:0l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:����������
dot/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_778512k
IdentityIdentitydot/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 Group_NN/StatefulPartitionedCall Group_NN/StatefulPartitionedCall2L
$Technique_NN/StatefulPartitionedCall$Technique_NN/StatefulPartitionedCall:YU
(
_output_shapes
:����������
)
_user_specified_nameinput_Technique:U Q
(
_output_shapes
:����������
%
_user_specified_nameinput_Group
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_779834

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_6_layer_call_fn_779783

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_778078o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
E__inference_output_NN_layer_call_and_return_conditional_losses_777761

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_779676

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_779696

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777902

inputs
dense_777871:	� 
dense_777873:  
dense_1_777876:  
dense_1_777878:  
dense_2_777881:  
dense_2_777883:  
dense_3_777886:  
dense_3_777888:  
dense_4_777891:  
dense_4_777893: "
output_nn_777896: 
output_nn_777898:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_777871dense_777873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_777677�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_777876dense_1_777878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_777694�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_777881dense_2_777883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_777711�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_777886dense_3_777888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_777728�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_777891dense_4_777893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_777745�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0output_nn_777896output_nn_777898*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_777761y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_output_NN_layer_call_and_return_conditional_losses_779873

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
C__inference_dense_5_layer_call_and_return_conditional_losses_778061

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_8_layer_call_fn_779823

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_778112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
-__inference_Technique_NN_layer_call_fn_779502

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�.
"__inference__traced_restore_780591
file_prefix0
assignvariableop_dense_kernel:	� +
assignvariableop_1_dense_bias: 3
!assignvariableop_2_dense_1_kernel:  -
assignvariableop_3_dense_1_bias: 3
!assignvariableop_4_dense_2_kernel:  -
assignvariableop_5_dense_2_bias: 3
!assignvariableop_6_dense_3_kernel:  -
assignvariableop_7_dense_3_bias: 3
!assignvariableop_8_dense_4_kernel:  -
assignvariableop_9_dense_4_bias: 8
&assignvariableop_10_output_nn_kernel_1: 2
$assignvariableop_11_output_nn_bias_1:5
"assignvariableop_12_dense_5_kernel:	�@.
 assignvariableop_13_dense_5_bias:@4
"assignvariableop_14_dense_6_kernel:@@.
 assignvariableop_15_dense_6_bias:@4
"assignvariableop_16_dense_7_kernel:@@.
 assignvariableop_17_dense_7_bias:@4
"assignvariableop_18_dense_8_kernel:@@.
 assignvariableop_19_dense_8_bias:@4
"assignvariableop_20_dense_9_kernel:@@.
 assignvariableop_21_dense_9_bias:@6
$assignvariableop_22_output_nn_kernel:@0
"assignvariableop_23_output_nn_bias:'
assignvariableop_24_iteration:	 +
!assignvariableop_25_learning_rate: :
'assignvariableop_26_adam_m_dense_kernel:	� :
'assignvariableop_27_adam_v_dense_kernel:	� 3
%assignvariableop_28_adam_m_dense_bias: 3
%assignvariableop_29_adam_v_dense_bias: ;
)assignvariableop_30_adam_m_dense_1_kernel:  ;
)assignvariableop_31_adam_v_dense_1_kernel:  5
'assignvariableop_32_adam_m_dense_1_bias: 5
'assignvariableop_33_adam_v_dense_1_bias: ;
)assignvariableop_34_adam_m_dense_2_kernel:  ;
)assignvariableop_35_adam_v_dense_2_kernel:  5
'assignvariableop_36_adam_m_dense_2_bias: 5
'assignvariableop_37_adam_v_dense_2_bias: ;
)assignvariableop_38_adam_m_dense_3_kernel:  ;
)assignvariableop_39_adam_v_dense_3_kernel:  5
'assignvariableop_40_adam_m_dense_3_bias: 5
'assignvariableop_41_adam_v_dense_3_bias: ;
)assignvariableop_42_adam_m_dense_4_kernel:  ;
)assignvariableop_43_adam_v_dense_4_kernel:  5
'assignvariableop_44_adam_m_dense_4_bias: 5
'assignvariableop_45_adam_v_dense_4_bias: ?
-assignvariableop_46_adam_m_output_nn_kernel_1: ?
-assignvariableop_47_adam_v_output_nn_kernel_1: 9
+assignvariableop_48_adam_m_output_nn_bias_1:9
+assignvariableop_49_adam_v_output_nn_bias_1:<
)assignvariableop_50_adam_m_dense_5_kernel:	�@<
)assignvariableop_51_adam_v_dense_5_kernel:	�@5
'assignvariableop_52_adam_m_dense_5_bias:@5
'assignvariableop_53_adam_v_dense_5_bias:@;
)assignvariableop_54_adam_m_dense_6_kernel:@@;
)assignvariableop_55_adam_v_dense_6_kernel:@@5
'assignvariableop_56_adam_m_dense_6_bias:@5
'assignvariableop_57_adam_v_dense_6_bias:@;
)assignvariableop_58_adam_m_dense_7_kernel:@@;
)assignvariableop_59_adam_v_dense_7_kernel:@@5
'assignvariableop_60_adam_m_dense_7_bias:@5
'assignvariableop_61_adam_v_dense_7_bias:@;
)assignvariableop_62_adam_m_dense_8_kernel:@@;
)assignvariableop_63_adam_v_dense_8_kernel:@@5
'assignvariableop_64_adam_m_dense_8_bias:@5
'assignvariableop_65_adam_v_dense_8_bias:@;
)assignvariableop_66_adam_m_dense_9_kernel:@@;
)assignvariableop_67_adam_v_dense_9_kernel:@@5
'assignvariableop_68_adam_m_dense_9_bias:@5
'assignvariableop_69_adam_v_dense_9_bias:@=
+assignvariableop_70_adam_m_output_nn_kernel:@=
+assignvariableop_71_adam_v_output_nn_kernel:@7
)assignvariableop_72_adam_m_output_nn_bias:7
)assignvariableop_73_adam_v_output_nn_bias:#
assignvariableop_74_total: #
assignvariableop_75_count: 
identity_77��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*�
value�B�MB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*�
value�B�MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_output_nn_kernel_1Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_output_nn_bias_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_5_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_5_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_6_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_6_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_7_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_7_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_8_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_8_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_9_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_9_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_output_nn_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_output_nn_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_iterationIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp!assignvariableop_25_learning_rateIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_m_dense_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_v_dense_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_m_dense_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_v_dense_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_m_dense_1_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_v_dense_1_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_m_dense_1_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_v_dense_1_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_dense_2_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_dense_2_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_dense_2_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_dense_2_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_dense_3_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_dense_3_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_m_dense_3_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_v_dense_3_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_m_dense_4_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_v_dense_4_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_m_dense_4_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_v_dense_4_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp-assignvariableop_46_adam_m_output_nn_kernel_1Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_v_output_nn_kernel_1Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_m_output_nn_bias_1Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_v_output_nn_bias_1Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_m_dense_5_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_v_dense_5_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_m_dense_5_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_v_dense_5_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_m_dense_6_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_v_dense_6_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_m_dense_6_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_v_dense_6_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_m_dense_7_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_v_dense_7_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_m_dense_7_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_v_dense_7_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_m_dense_8_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_v_dense_8_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_m_dense_8_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_v_dense_8_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_m_dense_9_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adam_v_dense_9_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adam_m_dense_9_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adam_v_dense_9_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp+assignvariableop_70_adam_m_output_nn_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_v_output_nn_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_m_output_nn_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp)assignvariableop_73_adam_v_output_nn_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpassignvariableop_74_totalIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOpassignvariableop_75_countIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_76Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_77IdentityIdentity_76:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_77Identity_77:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
P
$__inference_dot_layer_call_fn_779625
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_778512`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�

�
-__inference_Technique_NN_layer_call_fn_779531

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_779794

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_777728

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_dense_7_layer_call_fn_779803

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_778095o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�2
�	
D__inference_Group_NN_layer_call_and_return_conditional_losses_779473

inputs7
$dense_matmul_readvariableop_resource:	� 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource:  5
'dense_2_biasadd_readvariableop_resource: 8
&dense_3_matmul_readvariableop_resource:  5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: :
(output_nn_matmul_readvariableop_resource: 7
)output_nn_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� `
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output_NN/MatMulMatMuldense_4/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 output_NN/BiasAdd/ReadVariableOpReadVariableOp)output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_NN/BiasAddBiasAddoutput_NN/MatMul:product:0(output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentityoutput_NN/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_output_NN_layer_call_and_return_conditional_losses_779755

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_model1_layer_call_fn_779061
inputs_input_group
inputs_input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_input_groupinputs_input_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model1_layer_call_and_return_conditional_losses_778659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:`\
(
_output_shapes
:����������
0
_user_specified_nameinputs_input_technique:\ X
(
_output_shapes
:����������
,
_user_specified_nameinputs_input_group
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_779854

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�'
�
B__inference_model1_layer_call_and_return_conditional_losses_778515
input_group
input_technique"
group_nn_778436:	� 
group_nn_778438: !
group_nn_778440:  
group_nn_778442: !
group_nn_778444:  
group_nn_778446: !
group_nn_778448:  
group_nn_778450: !
group_nn_778452:  
group_nn_778454: !
group_nn_778456: 
group_nn_778458:&
technique_nn_778461:	�@!
technique_nn_778463:@%
technique_nn_778465:@@!
technique_nn_778467:@%
technique_nn_778469:@@!
technique_nn_778471:@%
technique_nn_778473:@@!
technique_nn_778475:@%
technique_nn_778477:@@!
technique_nn_778479:@%
technique_nn_778481:@!
technique_nn_778483:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinput_groupgroup_nn_778436group_nn_778438group_nn_778440group_nn_778442group_nn_778444group_nn_778446group_nn_778448group_nn_778450group_nn_778452group_nn_778454group_nn_778456group_nn_778458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777839�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinput_techniquetechnique_nn_778461technique_nn_778463technique_nn_778465technique_nn_778467technique_nn_778469technique_nn_778471technique_nn_778473technique_nn_778475technique_nn_778477technique_nn_778479technique_nn_778481technique_nn_778483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778223z
l2_normalize/SquareSquare)Group_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalizeMul)Group_NN/StatefulPartitionedCall:output:0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
l2_normalize_1/SquareSquare-Technique_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalize_1Mul-Technique_NN/StatefulPartitionedCall:output:0l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:����������
dot/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_778512k
IdentityIdentitydot/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 Group_NN/StatefulPartitionedCall Group_NN/StatefulPartitionedCall2L
$Technique_NN/StatefulPartitionedCall$Technique_NN/StatefulPartitionedCall:YU
(
_output_shapes
:����������
)
_user_specified_nameinput_Technique:U Q
(
_output_shapes
:����������
%
_user_specified_nameinput_Group
�
�
(__inference_dense_4_layer_call_fn_779725

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_777745o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
k
?__inference_dot_layer_call_and_return_conditional_losses_779637
inputs_0
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :q

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :u
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:���������y
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:���������R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::��l
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�2
�	
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779619

inputs9
&dense_5_matmul_readvariableop_resource:	�@5
'dense_5_biasadd_readvariableop_resource:@8
&dense_6_matmul_readvariableop_resource:@@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@@5
'dense_8_biasadd_readvariableop_resource:@8
&dense_9_matmul_readvariableop_resource:@@5
'dense_9_biasadd_readvariableop_resource:@:
(output_nn_matmul_readvariableop_resource:@7
)output_nn_biasadd_readvariableop_resource:
identity��dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0y
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_6/MatMulMatMuldense_5/BiasAdd:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
output_NN/MatMulMatMuldense_9/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 output_NN/BiasAdd/ReadVariableOpReadVariableOp)output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_NN/BiasAddBiasAddoutput_NN/MatMul:product:0(output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentityoutput_NN/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_Technique_NN_layer_call_fn_778313
dense_5_input
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_namedense_5_input
�

�
C__inference_dense_7_layer_call_and_return_conditional_losses_779814

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
C__inference_dense_3_layer_call_and_return_conditional_losses_779716

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
E__inference_output_NN_layer_call_and_return_conditional_losses_778145

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
'__inference_model1_layer_call_fn_778710
input_group
input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11:	�@

unknown_12:@

unknown_13:@@

unknown_14:@

unknown_15:@@

unknown_16:@

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_groupinput_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model1_layer_call_and_return_conditional_losses_778659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:YU
(
_output_shapes
:����������
)
_user_specified_nameinput_Technique:U Q
(
_output_shapes
:����������
%
_user_specified_nameinput_Group
� 
�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777768
dense_input
dense_777678:	� 
dense_777680:  
dense_1_777695:  
dense_1_777697:  
dense_2_777712:  
dense_2_777714:  
dense_3_777729:  
dense_3_777731:  
dense_4_777746:  
dense_4_777748: "
output_nn_777762: 
output_nn_777764:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_777678dense_777680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_777677�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_777695dense_1_777697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_777694�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_777712dense_2_777714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_777711�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_777729dense_3_777731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_777728�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_777746dense_4_777748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_777745�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0output_nn_777762output_nn_777764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_777761y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
(__inference_dense_3_layer_call_fn_779705

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_777728o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�'
�
B__inference_model1_layer_call_and_return_conditional_losses_778783

inputs
inputs_1"
group_nn_778717:	� 
group_nn_778719: !
group_nn_778721:  
group_nn_778723: !
group_nn_778725:  
group_nn_778727: !
group_nn_778729:  
group_nn_778731: !
group_nn_778733:  
group_nn_778735: !
group_nn_778737: 
group_nn_778739:&
technique_nn_778742:	�@!
technique_nn_778744:@%
technique_nn_778746:@@!
technique_nn_778748:@%
technique_nn_778750:@@!
technique_nn_778752:@%
technique_nn_778754:@@!
technique_nn_778756:@%
technique_nn_778758:@@!
technique_nn_778760:@%
technique_nn_778762:@!
technique_nn_778764:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinputsgroup_nn_778717group_nn_778719group_nn_778721group_nn_778723group_nn_778725group_nn_778727group_nn_778729group_nn_778731group_nn_778733group_nn_778735group_nn_778737group_nn_778739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777902�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinputs_1technique_nn_778742technique_nn_778744technique_nn_778746technique_nn_778748technique_nn_778750technique_nn_778752technique_nn_778754technique_nn_778756technique_nn_778758technique_nn_778760technique_nn_778762technique_nn_778764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778286z
l2_normalize/SquareSquare)Group_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalizeMul)Group_NN/StatefulPartitionedCall:output:0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
l2_normalize_1/SquareSquare-Technique_NN/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalize_1Mul-Technique_NN/StatefulPartitionedCall:output:0l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:����������
dot/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_dot_layer_call_and_return_conditional_losses_778512k
IdentityIdentitydot/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 Group_NN/StatefulPartitionedCall Group_NN/StatefulPartitionedCall2L
$Technique_NN/StatefulPartitionedCall$Technique_NN/StatefulPartitionedCall:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_6_layer_call_and_return_conditional_losses_778078

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
-__inference_Technique_NN_layer_call_fn_778250
dense_5_input
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778223o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_namedense_5_input
�
�
*__inference_output_NN_layer_call_fn_779863

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_778145o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_779685

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_777711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
ƌ
�
B__inference_model1_layer_call_and_return_conditional_losses_779221
inputs_input_group
inputs_input_technique@
-group_nn_dense_matmul_readvariableop_resource:	� <
.group_nn_dense_biasadd_readvariableop_resource: A
/group_nn_dense_1_matmul_readvariableop_resource:  >
0group_nn_dense_1_biasadd_readvariableop_resource: A
/group_nn_dense_2_matmul_readvariableop_resource:  >
0group_nn_dense_2_biasadd_readvariableop_resource: A
/group_nn_dense_3_matmul_readvariableop_resource:  >
0group_nn_dense_3_biasadd_readvariableop_resource: A
/group_nn_dense_4_matmul_readvariableop_resource:  >
0group_nn_dense_4_biasadd_readvariableop_resource: C
1group_nn_output_nn_matmul_readvariableop_resource: @
2group_nn_output_nn_biasadd_readvariableop_resource:F
3technique_nn_dense_5_matmul_readvariableop_resource:	�@B
4technique_nn_dense_5_biasadd_readvariableop_resource:@E
3technique_nn_dense_6_matmul_readvariableop_resource:@@B
4technique_nn_dense_6_biasadd_readvariableop_resource:@E
3technique_nn_dense_7_matmul_readvariableop_resource:@@B
4technique_nn_dense_7_biasadd_readvariableop_resource:@E
3technique_nn_dense_8_matmul_readvariableop_resource:@@B
4technique_nn_dense_8_biasadd_readvariableop_resource:@E
3technique_nn_dense_9_matmul_readvariableop_resource:@@B
4technique_nn_dense_9_biasadd_readvariableop_resource:@G
5technique_nn_output_nn_matmul_readvariableop_resource:@D
6technique_nn_output_nn_biasadd_readvariableop_resource:
identity��%Group_NN/dense/BiasAdd/ReadVariableOp�$Group_NN/dense/MatMul/ReadVariableOp�'Group_NN/dense_1/BiasAdd/ReadVariableOp�&Group_NN/dense_1/MatMul/ReadVariableOp�'Group_NN/dense_2/BiasAdd/ReadVariableOp�&Group_NN/dense_2/MatMul/ReadVariableOp�'Group_NN/dense_3/BiasAdd/ReadVariableOp�&Group_NN/dense_3/MatMul/ReadVariableOp�'Group_NN/dense_4/BiasAdd/ReadVariableOp�&Group_NN/dense_4/MatMul/ReadVariableOp�)Group_NN/output_NN/BiasAdd/ReadVariableOp�(Group_NN/output_NN/MatMul/ReadVariableOp�+Technique_NN/dense_5/BiasAdd/ReadVariableOp�*Technique_NN/dense_5/MatMul/ReadVariableOp�+Technique_NN/dense_6/BiasAdd/ReadVariableOp�*Technique_NN/dense_6/MatMul/ReadVariableOp�+Technique_NN/dense_7/BiasAdd/ReadVariableOp�*Technique_NN/dense_7/MatMul/ReadVariableOp�+Technique_NN/dense_8/BiasAdd/ReadVariableOp�*Technique_NN/dense_8/MatMul/ReadVariableOp�+Technique_NN/dense_9/BiasAdd/ReadVariableOp�*Technique_NN/dense_9/MatMul/ReadVariableOp�-Technique_NN/output_NN/BiasAdd/ReadVariableOp�,Technique_NN/output_NN/MatMul/ReadVariableOp�
$Group_NN/dense/MatMul/ReadVariableOpReadVariableOp-group_nn_dense_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
Group_NN/dense/MatMulMatMulinputs_input_group,Group_NN/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
%Group_NN/dense/BiasAdd/ReadVariableOpReadVariableOp.group_nn_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense/BiasAddBiasAddGroup_NN/dense/MatMul:product:0-Group_NN/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_1/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_1/MatMulMatMulGroup_NN/dense/BiasAdd:output:0.Group_NN/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_1/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_1/BiasAddBiasAdd!Group_NN/dense_1/MatMul:product:0/Group_NN/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_1/ReluRelu!Group_NN/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_2/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_2/MatMulMatMul#Group_NN/dense_1/Relu:activations:0.Group_NN/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_2/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_2/BiasAddBiasAdd!Group_NN/dense_2/MatMul:product:0/Group_NN/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_2/ReluRelu!Group_NN/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_3/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_3/MatMulMatMul#Group_NN/dense_2/Relu:activations:0.Group_NN/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_3/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_3/BiasAddBiasAdd!Group_NN/dense_3/MatMul:product:0/Group_NN/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_3/ReluRelu!Group_NN/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
&Group_NN/dense_4/MatMul/ReadVariableOpReadVariableOp/group_nn_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_4/MatMulMatMul#Group_NN/dense_3/Relu:activations:0.Group_NN/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_4/BiasAdd/ReadVariableOpReadVariableOp0group_nn_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_4/BiasAddBiasAdd!Group_NN/dense_4/MatMul:product:0/Group_NN/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
Group_NN/dense_4/ReluRelu!Group_NN/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp1group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Group_NN/output_NN/MatMulMatMul#Group_NN/dense_4/Relu:activations:00Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)Group_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp2group_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Group_NN/output_NN/BiasAddBiasAdd#Group_NN/output_NN/MatMul:product:01Group_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*Technique_NN/dense_5/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_5_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Technique_NN/dense_5/MatMulMatMulinputs_input_technique2Technique_NN/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_5/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_5/BiasAddBiasAdd%Technique_NN/dense_5/MatMul:product:03Technique_NN/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_6/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_6/MatMulMatMul%Technique_NN/dense_5/BiasAdd:output:02Technique_NN/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_6/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_6/BiasAddBiasAdd%Technique_NN/dense_6/MatMul:product:03Technique_NN/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_6/ReluRelu%Technique_NN/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_7/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_7/MatMulMatMul'Technique_NN/dense_6/Relu:activations:02Technique_NN/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_7/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_7/BiasAddBiasAdd%Technique_NN/dense_7/MatMul:product:03Technique_NN/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_7/ReluRelu%Technique_NN/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_8/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_8/MatMulMatMul'Technique_NN/dense_7/Relu:activations:02Technique_NN/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_8/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_8/BiasAddBiasAdd%Technique_NN/dense_8/MatMul:product:03Technique_NN/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_8/ReluRelu%Technique_NN/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*Technique_NN/dense_9/MatMul/ReadVariableOpReadVariableOp3technique_nn_dense_9_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_9/MatMulMatMul'Technique_NN/dense_8/Relu:activations:02Technique_NN/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_9/BiasAdd/ReadVariableOpReadVariableOp4technique_nn_dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_9/BiasAddBiasAdd%Technique_NN/dense_9/MatMul:product:03Technique_NN/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
Technique_NN/dense_9/ReluRelu%Technique_NN/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp5technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
Technique_NN/output_NN/MatMulMatMul'Technique_NN/dense_9/Relu:activations:04Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-Technique_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp6technique_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Technique_NN/output_NN/BiasAddBiasAdd'Technique_NN/output_NN/MatMul:product:05Technique_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
l2_normalize/SquareSquare#Group_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������d
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalizeMul#Group_NN/output_NN/BiasAdd:output:0l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:���������z
l2_normalize_1/SquareSquare'Technique_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������f
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
l2_normalize_1Mul'Technique_NN/output_NN/BiasAdd:output:0l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������T
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims
ExpandDimsl2_normalize:z:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������V
dot/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims_1
ExpandDimsl2_normalize_1:z:0dot/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������

dot/MatMulBatchMatMulV2dot/ExpandDims:output:0dot/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������Z
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
::��t
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
c
IdentityIdentitydot/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^Group_NN/dense/BiasAdd/ReadVariableOp%^Group_NN/dense/MatMul/ReadVariableOp(^Group_NN/dense_1/BiasAdd/ReadVariableOp'^Group_NN/dense_1/MatMul/ReadVariableOp(^Group_NN/dense_2/BiasAdd/ReadVariableOp'^Group_NN/dense_2/MatMul/ReadVariableOp(^Group_NN/dense_3/BiasAdd/ReadVariableOp'^Group_NN/dense_3/MatMul/ReadVariableOp(^Group_NN/dense_4/BiasAdd/ReadVariableOp'^Group_NN/dense_4/MatMul/ReadVariableOp*^Group_NN/output_NN/BiasAdd/ReadVariableOp)^Group_NN/output_NN/MatMul/ReadVariableOp,^Technique_NN/dense_5/BiasAdd/ReadVariableOp+^Technique_NN/dense_5/MatMul/ReadVariableOp,^Technique_NN/dense_6/BiasAdd/ReadVariableOp+^Technique_NN/dense_6/MatMul/ReadVariableOp,^Technique_NN/dense_7/BiasAdd/ReadVariableOp+^Technique_NN/dense_7/MatMul/ReadVariableOp,^Technique_NN/dense_8/BiasAdd/ReadVariableOp+^Technique_NN/dense_8/MatMul/ReadVariableOp,^Technique_NN/dense_9/BiasAdd/ReadVariableOp+^Technique_NN/dense_9/MatMul/ReadVariableOp.^Technique_NN/output_NN/BiasAdd/ReadVariableOp-^Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2N
%Group_NN/dense/BiasAdd/ReadVariableOp%Group_NN/dense/BiasAdd/ReadVariableOp2L
$Group_NN/dense/MatMul/ReadVariableOp$Group_NN/dense/MatMul/ReadVariableOp2R
'Group_NN/dense_1/BiasAdd/ReadVariableOp'Group_NN/dense_1/BiasAdd/ReadVariableOp2P
&Group_NN/dense_1/MatMul/ReadVariableOp&Group_NN/dense_1/MatMul/ReadVariableOp2R
'Group_NN/dense_2/BiasAdd/ReadVariableOp'Group_NN/dense_2/BiasAdd/ReadVariableOp2P
&Group_NN/dense_2/MatMul/ReadVariableOp&Group_NN/dense_2/MatMul/ReadVariableOp2R
'Group_NN/dense_3/BiasAdd/ReadVariableOp'Group_NN/dense_3/BiasAdd/ReadVariableOp2P
&Group_NN/dense_3/MatMul/ReadVariableOp&Group_NN/dense_3/MatMul/ReadVariableOp2R
'Group_NN/dense_4/BiasAdd/ReadVariableOp'Group_NN/dense_4/BiasAdd/ReadVariableOp2P
&Group_NN/dense_4/MatMul/ReadVariableOp&Group_NN/dense_4/MatMul/ReadVariableOp2V
)Group_NN/output_NN/BiasAdd/ReadVariableOp)Group_NN/output_NN/BiasAdd/ReadVariableOp2T
(Group_NN/output_NN/MatMul/ReadVariableOp(Group_NN/output_NN/MatMul/ReadVariableOp2Z
+Technique_NN/dense_5/BiasAdd/ReadVariableOp+Technique_NN/dense_5/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_5/MatMul/ReadVariableOp*Technique_NN/dense_5/MatMul/ReadVariableOp2Z
+Technique_NN/dense_6/BiasAdd/ReadVariableOp+Technique_NN/dense_6/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_6/MatMul/ReadVariableOp*Technique_NN/dense_6/MatMul/ReadVariableOp2Z
+Technique_NN/dense_7/BiasAdd/ReadVariableOp+Technique_NN/dense_7/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_7/MatMul/ReadVariableOp*Technique_NN/dense_7/MatMul/ReadVariableOp2Z
+Technique_NN/dense_8/BiasAdd/ReadVariableOp+Technique_NN/dense_8/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_8/MatMul/ReadVariableOp*Technique_NN/dense_8/MatMul/ReadVariableOp2Z
+Technique_NN/dense_9/BiasAdd/ReadVariableOp+Technique_NN/dense_9/BiasAdd/ReadVariableOp2X
*Technique_NN/dense_9/MatMul/ReadVariableOp*Technique_NN/dense_9/MatMul/ReadVariableOp2^
-Technique_NN/output_NN/BiasAdd/ReadVariableOp-Technique_NN/output_NN/BiasAdd/ReadVariableOp2\
,Technique_NN/output_NN/MatMul/ReadVariableOp,Technique_NN/output_NN/MatMul/ReadVariableOp:`\
(
_output_shapes
:����������
0
_user_specified_nameinputs_input_technique:\ X
(
_output_shapes
:����������
,
_user_specified_nameinputs_input_group
�2
�	
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779575

inputs9
&dense_5_matmul_readvariableop_resource:	�@5
'dense_5_biasadd_readvariableop_resource:@8
&dense_6_matmul_readvariableop_resource:@@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@@5
'dense_8_biasadd_readvariableop_resource:@8
&dense_9_matmul_readvariableop_resource:@@5
'dense_9_biasadd_readvariableop_resource:@:
(output_nn_matmul_readvariableop_resource:@7
)output_nn_biasadd_readvariableop_resource:
identity��dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0y
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_6/MatMulMatMuldense_5/BiasAdd:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
output_NN/MatMulMatMuldense_9/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 output_NN/BiasAdd/ReadVariableOpReadVariableOp)output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
output_NN/BiasAddBiasAddoutput_NN/MatMul:product:0(output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������i
IdentityIdentityoutput_NN/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_7_layer_call_and_return_conditional_losses_778095

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_5_layer_call_fn_779764

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_778061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
)__inference_Group_NN_layer_call_fn_777866
dense_input
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_Group_NN_layer_call_and_return_conditional_losses_777839o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_namedense_input
�
�
&__inference_dense_layer_call_fn_779646

inputs
unknown:	� 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_777677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778186
dense_5_input!
dense_5_778155:	�@
dense_5_778157:@ 
dense_6_778160:@@
dense_6_778162:@ 
dense_7_778165:@@
dense_7_778167:@ 
dense_8_778170:@@
dense_8_778172:@ 
dense_9_778175:@@
dense_9_778177:@"
output_nn_778180:@
output_nn_778182:
identity��dense_5/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCalldense_5_inputdense_5_778155dense_5_778157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_778061�
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_778160dense_6_778162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_778078�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_778165dense_7_778167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_778095�
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_778170dense_8_778172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_778112�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_778175dense_9_778177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_778129�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0output_nn_778180output_nn_778182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_output_NN_layer_call_and_return_conditional_losses_778145y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:W S
(
_output_shapes
:����������
'
_user_specified_namedense_5_input"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input_Group5
serving_default_input_Group:0����������
L
input_Technique9
!serving_default_input_Technique:0����������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
Group_NN
	Technique_NN

dot_product
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
*trace_0
+trace_1
,trace_2
-trace_32�
'__inference_model1_layer_call_fn_778710
'__inference_model1_layer_call_fn_778834
'__inference_model1_layer_call_fn_779061
'__inference_model1_layer_call_fn_779115�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z*trace_0z+trace_1z,trace_2z-trace_3
�
.trace_0
/trace_1
0trace_2
1trace_32�
B__inference_model1_layer_call_and_return_conditional_losses_778515
B__inference_model1_layer_call_and_return_conditional_losses_778585
B__inference_model1_layer_call_and_return_conditional_losses_779221
B__inference_model1_layer_call_and_return_conditional_losses_779327�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z.trace_0z/trace_1z0trace_2z1trace_3
�B�
!__inference__wrapped_model_777663input_Groupinput_Technique"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
2layer_with_weights-0
2layer-0
3layer_with_weights-1
3layer-1
4layer_with_weights-2
4layer-2
5layer_with_weights-3
5layer-3
6layer_with_weights-4
6layer-4
7layer_with_weights-5
7layer-5
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
>layer_with_weights-0
>layer-0
?layer_with_weights-1
?layer-1
@layer_with_weights-2
@layer-2
Alayer_with_weights-3
Alayer-3
Blayer_with_weights-4
Blayer-4
Clayer_with_weights-5
Clayer-5
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
�
P
_variables
Q_iterations
R_learning_rate
S_index_dict
T
_momentums
U_velocities
V_update_step_xla"
experimentalOptimizer
,
Wserving_default"
signature_map
:	� 2dense/kernel
: 2
dense/bias
 :  2dense_1/kernel
: 2dense_1/bias
 :  2dense_2/kernel
: 2dense_2/bias
 :  2dense_3/kernel
: 2dense_3/bias
 :  2dense_4/kernel
: 2dense_4/bias
":  2output_NN/kernel
:2output_NN/bias
!:	�@2dense_5/kernel
:@2dense_5/bias
 :@@2dense_6/kernel
:@2dense_6/bias
 :@@2dense_7/kernel
:@2dense_7/bias
 :@@2dense_8/kernel
:@2dense_8/bias
 :@@2dense_9/kernel
:@2dense_9/bias
": @2output_NN/kernel
:2output_NN/bias
 "
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model1_layer_call_fn_778710input_Groupinput_Technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
'__inference_model1_layer_call_fn_778834input_Groupinput_Technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
'__inference_model1_layer_call_fn_779061inputs_input_groupinputs_input_technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
'__inference_model1_layer_call_fn_779115inputs_input_groupinputs_input_technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_model1_layer_call_and_return_conditional_losses_778515input_Groupinput_Technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_model1_layer_call_and_return_conditional_losses_778585input_Groupinput_Technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_model1_layer_call_and_return_conditional_losses_779221inputs_input_groupinputs_input_technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_model1_layer_call_and_return_conditional_losses_779327inputs_input_groupinputs_input_technique"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
)__inference_Group_NN_layer_call_fn_777866
)__inference_Group_NN_layer_call_fn_777929
)__inference_Group_NN_layer_call_fn_779356
)__inference_Group_NN_layer_call_fn_779385�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777768
D__inference_Group_NN_layer_call_and_return_conditional_losses_777802
D__inference_Group_NN_layer_call_and_return_conditional_losses_779429
D__inference_Group_NN_layer_call_and_return_conditional_losses_779473�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
-__inference_Technique_NN_layer_call_fn_778250
-__inference_Technique_NN_layer_call_fn_778313
-__inference_Technique_NN_layer_call_fn_779502
-__inference_Technique_NN_layer_call_fn_779531�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778152
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778186
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779575
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779619�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_dot_layer_call_fn_779625�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_dot_layer_call_and_return_conditional_losses_779637�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
Q0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_779007input_Groupinput_Technique"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_779646�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_779656�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_779665�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_779676�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_779685�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_779696�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_3_layer_call_fn_779705�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_779716�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_4_layer_call_fn_779725�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_4_layer_call_and_return_conditional_losses_779736�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_output_NN_layer_call_fn_779745�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_output_NN_layer_call_and_return_conditional_losses_779755�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
J
20
31
42
53
64
75"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_Group_NN_layer_call_fn_777866dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Group_NN_layer_call_fn_777929dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Group_NN_layer_call_fn_779356inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_Group_NN_layer_call_fn_779385inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777768dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Group_NN_layer_call_and_return_conditional_losses_777802dense_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Group_NN_layer_call_and_return_conditional_losses_779429inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_Group_NN_layer_call_and_return_conditional_losses_779473inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_5_layer_call_fn_779764�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_5_layer_call_and_return_conditional_losses_779774�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_6_layer_call_fn_779783�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_6_layer_call_and_return_conditional_losses_779794�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_7_layer_call_fn_779803�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_7_layer_call_and_return_conditional_losses_779814�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_8_layer_call_fn_779823�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_8_layer_call_and_return_conditional_losses_779834�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_9_layer_call_fn_779843�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_9_layer_call_and_return_conditional_losses_779854�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_output_NN_layer_call_fn_779863�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_output_NN_layer_call_and_return_conditional_losses_779873�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
J
>0
?1
@2
A3
B4
C5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_Technique_NN_layer_call_fn_778250dense_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_Technique_NN_layer_call_fn_778313dense_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_Technique_NN_layer_call_fn_779502inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_Technique_NN_layer_call_fn_779531inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778152dense_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778186dense_5_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779575inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779619inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_dot_layer_call_fn_779625inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_dot_layer_call_and_return_conditional_losses_779637inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
$:"	� 2Adam/m/dense/kernel
$:"	� 2Adam/v/dense/kernel
: 2Adam/m/dense/bias
: 2Adam/v/dense/bias
%:#  2Adam/m/dense_1/kernel
%:#  2Adam/v/dense_1/kernel
: 2Adam/m/dense_1/bias
: 2Adam/v/dense_1/bias
%:#  2Adam/m/dense_2/kernel
%:#  2Adam/v/dense_2/kernel
: 2Adam/m/dense_2/bias
: 2Adam/v/dense_2/bias
%:#  2Adam/m/dense_3/kernel
%:#  2Adam/v/dense_3/kernel
: 2Adam/m/dense_3/bias
: 2Adam/v/dense_3/bias
%:#  2Adam/m/dense_4/kernel
%:#  2Adam/v/dense_4/kernel
: 2Adam/m/dense_4/bias
: 2Adam/v/dense_4/bias
':% 2Adam/m/output_NN/kernel
':% 2Adam/v/output_NN/kernel
!:2Adam/m/output_NN/bias
!:2Adam/v/output_NN/bias
&:$	�@2Adam/m/dense_5/kernel
&:$	�@2Adam/v/dense_5/kernel
:@2Adam/m/dense_5/bias
:@2Adam/v/dense_5/bias
%:#@@2Adam/m/dense_6/kernel
%:#@@2Adam/v/dense_6/kernel
:@2Adam/m/dense_6/bias
:@2Adam/v/dense_6/bias
%:#@@2Adam/m/dense_7/kernel
%:#@@2Adam/v/dense_7/kernel
:@2Adam/m/dense_7/bias
:@2Adam/v/dense_7/bias
%:#@@2Adam/m/dense_8/kernel
%:#@@2Adam/v/dense_8/kernel
:@2Adam/m/dense_8/bias
:@2Adam/v/dense_8/bias
%:#@@2Adam/m/dense_9/kernel
%:#@@2Adam/v/dense_9/kernel
:@2Adam/m/dense_9/bias
:@2Adam/v/dense_9/bias
':%@2Adam/m/output_NN/kernel
':%@2Adam/v/output_NN/kernel
!:2Adam/m/output_NN/bias
!:2Adam/v/output_NN/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_779646inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_779656inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_779665inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_779676inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_779685inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_779696inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_779705inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_779716inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_4_layer_call_fn_779725inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_4_layer_call_and_return_conditional_losses_779736inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_output_NN_layer_call_fn_779745inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_output_NN_layer_call_and_return_conditional_losses_779755inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_5_layer_call_fn_779764inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_5_layer_call_and_return_conditional_losses_779774inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_6_layer_call_fn_779783inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_6_layer_call_and_return_conditional_losses_779794inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_7_layer_call_fn_779803inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_7_layer_call_and_return_conditional_losses_779814inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_8_layer_call_fn_779823inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_8_layer_call_and_return_conditional_losses_779834inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_9_layer_call_fn_779843inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_9_layer_call_and_return_conditional_losses_779854inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_output_NN_layer_call_fn_779863inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_output_NN_layer_call_and_return_conditional_losses_779873inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
D__inference_Group_NN_layer_call_and_return_conditional_losses_777768{=�:
3�0
&�#
dense_input����������
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_Group_NN_layer_call_and_return_conditional_losses_777802{=�:
3�0
&�#
dense_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
D__inference_Group_NN_layer_call_and_return_conditional_losses_779429v8�5
.�+
!�
inputs����������
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_Group_NN_layer_call_and_return_conditional_losses_779473v8�5
.�+
!�
inputs����������
p 

 
� ",�)
"�
tensor_0���������
� �
)__inference_Group_NN_layer_call_fn_777866p=�:
3�0
&�#
dense_input����������
p

 
� "!�
unknown����������
)__inference_Group_NN_layer_call_fn_777929p=�:
3�0
&�#
dense_input����������
p 

 
� "!�
unknown����������
)__inference_Group_NN_layer_call_fn_779356k8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
)__inference_Group_NN_layer_call_fn_779385k8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778152} !"#$?�<
5�2
(�%
dense_5_input����������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_Technique_NN_layer_call_and_return_conditional_losses_778186} !"#$?�<
5�2
(�%
dense_5_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779575v !"#$8�5
.�+
!�
inputs����������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_Technique_NN_layer_call_and_return_conditional_losses_779619v !"#$8�5
.�+
!�
inputs����������
p 

 
� ",�)
"�
tensor_0���������
� �
-__inference_Technique_NN_layer_call_fn_778250r !"#$?�<
5�2
(�%
dense_5_input����������
p

 
� "!�
unknown����������
-__inference_Technique_NN_layer_call_fn_778313r !"#$?�<
5�2
(�%
dense_5_input����������
p 

 
� "!�
unknown����������
-__inference_Technique_NN_layer_call_fn_779502k !"#$8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
-__inference_Technique_NN_layer_call_fn_779531k !"#$8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
!__inference__wrapped_model_777663� !"#$���
~�{
y�v
5
input_Group&�#
input_Group����������
=
input_Technique*�'
input_Technique����������
� "3�0
.
output_1"�
output_1����������
C__inference_dense_1_layer_call_and_return_conditional_losses_779676c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense_1_layer_call_fn_779665X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
C__inference_dense_2_layer_call_and_return_conditional_losses_779696c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense_2_layer_call_fn_779685X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
C__inference_dense_3_layer_call_and_return_conditional_losses_779716c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense_3_layer_call_fn_779705X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
C__inference_dense_4_layer_call_and_return_conditional_losses_779736c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
(__inference_dense_4_layer_call_fn_779725X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
C__inference_dense_5_layer_call_and_return_conditional_losses_779774d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_5_layer_call_fn_779764Y0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
C__inference_dense_6_layer_call_and_return_conditional_losses_779794c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_6_layer_call_fn_779783X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_dense_7_layer_call_and_return_conditional_losses_779814c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_7_layer_call_fn_779803X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_dense_8_layer_call_and_return_conditional_losses_779834c /�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_8_layer_call_fn_779823X /�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_dense_9_layer_call_and_return_conditional_losses_779854c!"/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
(__inference_dense_9_layer_call_fn_779843X!"/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
A__inference_dense_layer_call_and_return_conditional_losses_779656d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0��������� 
� �
&__inference_dense_layer_call_fn_779646Y0�-
&�#
!�
inputs����������
� "!�
unknown��������� �
?__inference_dot_layer_call_and_return_conditional_losses_779637�Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� ",�)
"�
tensor_0���������
� �
$__inference_dot_layer_call_fn_779625Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� "!�
unknown����������
B__inference_model1_layer_call_and_return_conditional_losses_778515� !"#$���
~�{
y�v
5
input_Group&�#
input_Group����������
=
input_Technique*�'
input_Technique����������
�

trainingp",�)
"�
tensor_0���������
� �
B__inference_model1_layer_call_and_return_conditional_losses_778585� !"#$���
~�{
y�v
5
input_Group&�#
input_Group����������
=
input_Technique*�'
input_Technique����������
�

trainingp ",�)
"�
tensor_0���������
� �
B__inference_model1_layer_call_and_return_conditional_losses_779221� !"#$���
���
���
<
input_Group-�*
inputs_input_group����������
D
input_Technique1�.
inputs_input_technique����������
�

trainingp",�)
"�
tensor_0���������
� �
B__inference_model1_layer_call_and_return_conditional_losses_779327� !"#$���
���
���
<
input_Group-�*
inputs_input_group����������
D
input_Technique1�.
inputs_input_technique����������
�

trainingp ",�)
"�
tensor_0���������
� �
'__inference_model1_layer_call_fn_778710� !"#$���
~�{
y�v
5
input_Group&�#
input_Group����������
=
input_Technique*�'
input_Technique����������
�

trainingp"!�
unknown����������
'__inference_model1_layer_call_fn_778834� !"#$���
~�{
y�v
5
input_Group&�#
input_Group����������
=
input_Technique*�'
input_Technique����������
�

trainingp "!�
unknown����������
'__inference_model1_layer_call_fn_779061� !"#$���
���
���
<
input_Group-�*
inputs_input_group����������
D
input_Technique1�.
inputs_input_technique����������
�

trainingp"!�
unknown����������
'__inference_model1_layer_call_fn_779115� !"#$���
���
���
<
input_Group-�*
inputs_input_group����������
D
input_Technique1�.
inputs_input_technique����������
�

trainingp "!�
unknown����������
E__inference_output_NN_layer_call_and_return_conditional_losses_779755c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
E__inference_output_NN_layer_call_and_return_conditional_losses_779873c#$/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
*__inference_output_NN_layer_call_fn_779745X/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
*__inference_output_NN_layer_call_fn_779863X#$/�,
%�"
 �
inputs���������@
� "!�
unknown����������
$__inference_signature_wrapper_779007� !"#$���
� 
y�v
5
input_Group&�#
input_group����������
=
input_Technique*�'
input_technique����������"3�0
.
output_1"�
output_1���������