��
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
�
Adam/v/dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_29/bias
y
(Adam/v/dense_29/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_29/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_29/bias
y
(Adam/m/dense_29/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_29/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_29/kernel
�
*Adam/v/dense_29/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_29/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_29/kernel
�
*Adam/m/dense_29/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_29/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_28/bias
y
(Adam/v/dense_28/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_28/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_28/bias
y
(Adam/m/dense_28/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_28/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_28/kernel
�
*Adam/v/dense_28/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_28/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_28/kernel
�
*Adam/m/dense_28/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_28/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_27/bias
y
(Adam/v/dense_27/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_27/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_27/bias
y
(Adam/m/dense_27/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_27/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_27/kernel
�
*Adam/v/dense_27/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_27/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_27/kernel
�
*Adam/m/dense_27/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_27/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_26/bias
y
(Adam/v/dense_26/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_26/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_26/bias
y
(Adam/m/dense_26/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_26/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_26/kernel
�
*Adam/v/dense_26/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_26/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_26/kernel
�
*Adam/m/dense_26/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_26/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_25/bias
y
(Adam/v/dense_25/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_25/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_25/bias
y
(Adam/m/dense_25/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_25/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/v/dense_25/kernel
�
*Adam/v/dense_25/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_25/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/m/dense_25/kernel
�
*Adam/m/dense_25/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_25/kernel*
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
�
Adam/v/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_24/bias
y
(Adam/v/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_24/bias
y
(Adam/m/dense_24/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_24/kernel
�
*Adam/v/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_24/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_24/kernel
�
*Adam/m/dense_24/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_24/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_23/bias
y
(Adam/v/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_23/bias
y
(Adam/m/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_23/kernel
�
*Adam/v/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_23/kernel
�
*Adam/m/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_22/bias
y
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_22/bias
y
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_22/kernel
�
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_22/kernel
�
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_21/bias
y
(Adam/v/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_21/bias
y
(Adam/m/dense_21/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_21/kernel
�
*Adam/v/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_21/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_21/kernel
�
*Adam/m/dense_21/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_21/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_20/bias
y
(Adam/v/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_20/bias
y
(Adam/m/dense_20/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/v/dense_20/kernel
�
*Adam/v/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_20/kernel*
_output_shapes
:	� *
dtype0
�
Adam/m/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/m/dense_20/kernel
�
*Adam/m/dense_20/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_20/kernel*
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
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:@*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:@@*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:@*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:@@*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:@*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:@@*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:@*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:@@*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:@*
dtype0
{
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_25/kernel
t
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
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
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
: *
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:  *
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
: *
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:  *
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
: *
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:  *
dtype0
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
: *
dtype0
z
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_21/kernel
s
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes

:  *
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
: *
dtype0
{
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� * 
shared_namedense_20/kernel
t
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
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
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_Groupserving_default_input_Techniquedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasoutput_NN/kernel_1output_NN/bias_1dense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasoutput_NN/kerneloutput_NN/bias*%
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
GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2340829

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ݒ
valueҒBΒ Bƒ
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
OI
VARIABLE_VALUEdense_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_20/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_21/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_21/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_22/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_22/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_23/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_23/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_24/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_24/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEoutput_NN/kernel_1'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEoutput_NN/bias_1'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_25/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_25/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_26/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_26/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_27/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_27/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_28/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_28/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_29/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_29/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
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
a[
VARIABLE_VALUEAdam/m/dense_20/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_20/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_20/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_20/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_21/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_21/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_21/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_21/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_22/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_22/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_22/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_22/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_23/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_23/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_23/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_23/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_24/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_24/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_24/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_24/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/output_NN/kernel_12optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/output_NN/kernel_12optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/output_NN/bias_12optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/output_NN/bias_12optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_25/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_25/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_25/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_25/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_26/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_26/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_26/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_26/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_27/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_27/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_27/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_27/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_28/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_28/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_28/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_28/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_29/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_29/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_29/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_29/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasoutput_NN/kernel_1output_NN/bias_1dense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasoutput_NN/kerneloutput_NN/bias	iterationlearning_rateAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biasAdam/m/output_NN/kernel_1Adam/v/output_NN/kernel_1Adam/m/output_NN/bias_1Adam/v/output_NN/bias_1Adam/m/dense_25/kernelAdam/v/dense_25/kernelAdam/m/dense_25/biasAdam/v/dense_25/biasAdam/m/dense_26/kernelAdam/v/dense_26/kernelAdam/m/dense_26/biasAdam/v/dense_26/biasAdam/m/dense_27/kernelAdam/v/dense_27/kernelAdam/m/dense_27/biasAdam/v/dense_27/biasAdam/m/dense_28/kernelAdam/v/dense_28/kernelAdam/m/dense_28/biasAdam/v/dense_28/biasAdam/m/dense_29/kernelAdam/v/dense_29/kernelAdam/m/dense_29/biasAdam/v/dense_29/biasAdam/m/output_NN/kernelAdam/v/output_NN/kernelAdam/m/output_NN/biasAdam/v/output_NN/biastotalcountConst*Y
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
GPU 2J 8� *)
f$R"
 __inference__traced_save_2342175
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasoutput_NN/kernel_1output_NN/bias_1dense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/biasoutput_NN/kerneloutput_NN/bias	iterationlearning_rateAdam/m/dense_20/kernelAdam/v/dense_20/kernelAdam/m/dense_20/biasAdam/v/dense_20/biasAdam/m/dense_21/kernelAdam/v/dense_21/kernelAdam/m/dense_21/biasAdam/v/dense_21/biasAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biasAdam/m/dense_24/kernelAdam/v/dense_24/kernelAdam/m/dense_24/biasAdam/v/dense_24/biasAdam/m/output_NN/kernel_1Adam/v/output_NN/kernel_1Adam/m/output_NN/bias_1Adam/v/output_NN/bias_1Adam/m/dense_25/kernelAdam/v/dense_25/kernelAdam/m/dense_25/biasAdam/v/dense_25/biasAdam/m/dense_26/kernelAdam/v/dense_26/kernelAdam/m/dense_26/biasAdam/v/dense_26/biasAdam/m/dense_27/kernelAdam/v/dense_27/kernelAdam/m/dense_27/biasAdam/v/dense_27/biasAdam/m/dense_28/kernelAdam/v/dense_28/kernelAdam/m/dense_28/biasAdam/v/dense_28/biasAdam/m/dense_29/kernelAdam/v/dense_29/kernelAdam/m/dense_29/biasAdam/v/dense_29/biasAdam/m/output_NN/kernelAdam/v/output_NN/kernelAdam/m/output_NN/biasAdam/v/output_NN/biastotalcount*X
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
GPU 2J 8� *,
f'R%
#__inference__traced_restore_2342413ѓ
� 
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340108

inputs#
dense_25_2340077:	�@
dense_25_2340079:@"
dense_26_2340082:@@
dense_26_2340084:@"
dense_27_2340087:@@
dense_27_2340089:@"
dense_28_2340092:@@
dense_28_2340094:@"
dense_29_2340097:@@
dense_29_2340099:@#
output_nn_2340102:@
output_nn_2340104:
identity�� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25_2340077dense_25_2340079*
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
GPU 2J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2339883�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_2340082dense_26_2340084*
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
GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2339900�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2340087dense_27_2340089*
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
GPU 2J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2339917�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_2340092dense_28_2340094*
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
GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2339934�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2340097dense_29_2340099*
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
GPU 2J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_2339951�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0output_nn_2340102output_nn_2340104*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339967y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_20_layer_call_fn_2341468

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
GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2339499o
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

�
E__inference_dense_23_layer_call_and_return_conditional_losses_2341538

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
+__inference_output_NN_layer_call_fn_2341567

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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339583o
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
�3
�	
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341441

inputs:
'dense_25_matmul_readvariableop_resource:	�@6
(dense_25_biasadd_readvariableop_resource:@9
'dense_26_matmul_readvariableop_resource:@@6
(dense_26_biasadd_readvariableop_resource:@9
'dense_27_matmul_readvariableop_resource:@@6
(dense_27_biasadd_readvariableop_resource:@9
'dense_28_matmul_readvariableop_resource:@@6
(dense_28_biasadd_readvariableop_resource:@9
'dense_29_matmul_readvariableop_resource:@@6
(dense_29_biasadd_readvariableop_resource:@:
(output_nn_matmul_readvariableop_resource:@7
)output_nn_biasadd_readvariableop_resource:
identity��dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0{
dense_25/MatMulMatMulinputs&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_26/MatMulMatMuldense_25/BiasAdd:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
output_NN/MatMulMatMuldense_29/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model1_2_layer_call_fn_2340937
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
GPU 2J 8� *N
fIRG
E__inference_model1_2_layer_call_and_return_conditional_losses_2340605o
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
�
�
E__inference_model1_2_layer_call_and_return_conditional_losses_2341043
inputs_input_group
inputs_input_techniqueC
0group_nn_dense_20_matmul_readvariableop_resource:	� ?
1group_nn_dense_20_biasadd_readvariableop_resource: B
0group_nn_dense_21_matmul_readvariableop_resource:  ?
1group_nn_dense_21_biasadd_readvariableop_resource: B
0group_nn_dense_22_matmul_readvariableop_resource:  ?
1group_nn_dense_22_biasadd_readvariableop_resource: B
0group_nn_dense_23_matmul_readvariableop_resource:  ?
1group_nn_dense_23_biasadd_readvariableop_resource: B
0group_nn_dense_24_matmul_readvariableop_resource:  ?
1group_nn_dense_24_biasadd_readvariableop_resource: C
1group_nn_output_nn_matmul_readvariableop_resource: @
2group_nn_output_nn_biasadd_readvariableop_resource:G
4technique_nn_dense_25_matmul_readvariableop_resource:	�@C
5technique_nn_dense_25_biasadd_readvariableop_resource:@F
4technique_nn_dense_26_matmul_readvariableop_resource:@@C
5technique_nn_dense_26_biasadd_readvariableop_resource:@F
4technique_nn_dense_27_matmul_readvariableop_resource:@@C
5technique_nn_dense_27_biasadd_readvariableop_resource:@F
4technique_nn_dense_28_matmul_readvariableop_resource:@@C
5technique_nn_dense_28_biasadd_readvariableop_resource:@F
4technique_nn_dense_29_matmul_readvariableop_resource:@@C
5technique_nn_dense_29_biasadd_readvariableop_resource:@G
5technique_nn_output_nn_matmul_readvariableop_resource:@D
6technique_nn_output_nn_biasadd_readvariableop_resource:
identity��(Group_NN/dense_20/BiasAdd/ReadVariableOp�'Group_NN/dense_20/MatMul/ReadVariableOp�(Group_NN/dense_21/BiasAdd/ReadVariableOp�'Group_NN/dense_21/MatMul/ReadVariableOp�(Group_NN/dense_22/BiasAdd/ReadVariableOp�'Group_NN/dense_22/MatMul/ReadVariableOp�(Group_NN/dense_23/BiasAdd/ReadVariableOp�'Group_NN/dense_23/MatMul/ReadVariableOp�(Group_NN/dense_24/BiasAdd/ReadVariableOp�'Group_NN/dense_24/MatMul/ReadVariableOp�)Group_NN/output_NN/BiasAdd/ReadVariableOp�(Group_NN/output_NN/MatMul/ReadVariableOp�,Technique_NN/dense_25/BiasAdd/ReadVariableOp�+Technique_NN/dense_25/MatMul/ReadVariableOp�,Technique_NN/dense_26/BiasAdd/ReadVariableOp�+Technique_NN/dense_26/MatMul/ReadVariableOp�,Technique_NN/dense_27/BiasAdd/ReadVariableOp�+Technique_NN/dense_27/MatMul/ReadVariableOp�,Technique_NN/dense_28/BiasAdd/ReadVariableOp�+Technique_NN/dense_28/MatMul/ReadVariableOp�,Technique_NN/dense_29/BiasAdd/ReadVariableOp�+Technique_NN/dense_29/MatMul/ReadVariableOp�-Technique_NN/output_NN/BiasAdd/ReadVariableOp�,Technique_NN/output_NN/MatMul/ReadVariableOp�
'Group_NN/dense_20/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_20_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
Group_NN/dense_20/MatMulMatMulinputs_input_group/Group_NN/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_20/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_20/BiasAddBiasAdd"Group_NN/dense_20/MatMul:product:00Group_NN/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_21/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_21_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_21/MatMulMatMul"Group_NN/dense_20/BiasAdd:output:0/Group_NN/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_21/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_21/BiasAddBiasAdd"Group_NN/dense_21/MatMul:product:00Group_NN/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_21/ReluRelu"Group_NN/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_22/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_22_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_22/MatMulMatMul$Group_NN/dense_21/Relu:activations:0/Group_NN/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_22/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_22/BiasAddBiasAdd"Group_NN/dense_22/MatMul:product:00Group_NN/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_22/ReluRelu"Group_NN/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_23/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_23/MatMulMatMul$Group_NN/dense_22/Relu:activations:0/Group_NN/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_23/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_23/BiasAddBiasAdd"Group_NN/dense_23/MatMul:product:00Group_NN/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_23/ReluRelu"Group_NN/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_24/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_24_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_24/MatMulMatMul$Group_NN/dense_23/Relu:activations:0/Group_NN/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_24/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_24/BiasAddBiasAdd"Group_NN/dense_24/MatMul:product:00Group_NN/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_24/ReluRelu"Group_NN/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp1group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Group_NN/output_NN/MatMulMatMul$Group_NN/dense_24/Relu:activations:00Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
+Technique_NN/dense_25/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_25_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Technique_NN/dense_25/MatMulMatMulinputs_input_technique3Technique_NN/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_25/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_25/BiasAddBiasAdd&Technique_NN/dense_25/MatMul:product:04Technique_NN/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_26/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_26/MatMulMatMul&Technique_NN/dense_25/BiasAdd:output:03Technique_NN/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_26/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_26/BiasAddBiasAdd&Technique_NN/dense_26/MatMul:product:04Technique_NN/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_26/ReluRelu&Technique_NN/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_27/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_27/MatMulMatMul(Technique_NN/dense_26/Relu:activations:03Technique_NN/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_27/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_27/BiasAddBiasAdd&Technique_NN/dense_27/MatMul:product:04Technique_NN/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_27/ReluRelu&Technique_NN/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_28/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_28/MatMulMatMul(Technique_NN/dense_27/Relu:activations:03Technique_NN/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_28/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_28/BiasAddBiasAdd&Technique_NN/dense_28/MatMul:product:04Technique_NN/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_28/ReluRelu&Technique_NN/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_29/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_29_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_29/MatMulMatMul(Technique_NN/dense_28/Relu:activations:03Technique_NN/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_29/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_29/BiasAddBiasAdd&Technique_NN/dense_29/MatMul:product:04Technique_NN/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_29/ReluRelu&Technique_NN/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp5technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
Technique_NN/output_NN/MatMulMatMul(Technique_NN/dense_29/Relu:activations:04Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
:���������V
dot_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims
ExpandDimsl2_normalize:z:0dot_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������X
dot_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims_1
ExpandDimsl2_normalize_1:z:0dot_2/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
dot_2/MatMulBatchMatMulV2dot_2/ExpandDims:output:0dot_2/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������^
dot_2/ShapeShapedot_2/MatMul:output:0*
T0*
_output_shapes
::��x
dot_2/SqueezeSqueezedot_2/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
e
IdentityIdentitydot_2/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^Group_NN/dense_20/BiasAdd/ReadVariableOp(^Group_NN/dense_20/MatMul/ReadVariableOp)^Group_NN/dense_21/BiasAdd/ReadVariableOp(^Group_NN/dense_21/MatMul/ReadVariableOp)^Group_NN/dense_22/BiasAdd/ReadVariableOp(^Group_NN/dense_22/MatMul/ReadVariableOp)^Group_NN/dense_23/BiasAdd/ReadVariableOp(^Group_NN/dense_23/MatMul/ReadVariableOp)^Group_NN/dense_24/BiasAdd/ReadVariableOp(^Group_NN/dense_24/MatMul/ReadVariableOp*^Group_NN/output_NN/BiasAdd/ReadVariableOp)^Group_NN/output_NN/MatMul/ReadVariableOp-^Technique_NN/dense_25/BiasAdd/ReadVariableOp,^Technique_NN/dense_25/MatMul/ReadVariableOp-^Technique_NN/dense_26/BiasAdd/ReadVariableOp,^Technique_NN/dense_26/MatMul/ReadVariableOp-^Technique_NN/dense_27/BiasAdd/ReadVariableOp,^Technique_NN/dense_27/MatMul/ReadVariableOp-^Technique_NN/dense_28/BiasAdd/ReadVariableOp,^Technique_NN/dense_28/MatMul/ReadVariableOp-^Technique_NN/dense_29/BiasAdd/ReadVariableOp,^Technique_NN/dense_29/MatMul/ReadVariableOp.^Technique_NN/output_NN/BiasAdd/ReadVariableOp-^Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2T
(Group_NN/dense_20/BiasAdd/ReadVariableOp(Group_NN/dense_20/BiasAdd/ReadVariableOp2R
'Group_NN/dense_20/MatMul/ReadVariableOp'Group_NN/dense_20/MatMul/ReadVariableOp2T
(Group_NN/dense_21/BiasAdd/ReadVariableOp(Group_NN/dense_21/BiasAdd/ReadVariableOp2R
'Group_NN/dense_21/MatMul/ReadVariableOp'Group_NN/dense_21/MatMul/ReadVariableOp2T
(Group_NN/dense_22/BiasAdd/ReadVariableOp(Group_NN/dense_22/BiasAdd/ReadVariableOp2R
'Group_NN/dense_22/MatMul/ReadVariableOp'Group_NN/dense_22/MatMul/ReadVariableOp2T
(Group_NN/dense_23/BiasAdd/ReadVariableOp(Group_NN/dense_23/BiasAdd/ReadVariableOp2R
'Group_NN/dense_23/MatMul/ReadVariableOp'Group_NN/dense_23/MatMul/ReadVariableOp2T
(Group_NN/dense_24/BiasAdd/ReadVariableOp(Group_NN/dense_24/BiasAdd/ReadVariableOp2R
'Group_NN/dense_24/MatMul/ReadVariableOp'Group_NN/dense_24/MatMul/ReadVariableOp2V
)Group_NN/output_NN/BiasAdd/ReadVariableOp)Group_NN/output_NN/BiasAdd/ReadVariableOp2T
(Group_NN/output_NN/MatMul/ReadVariableOp(Group_NN/output_NN/MatMul/ReadVariableOp2\
,Technique_NN/dense_25/BiasAdd/ReadVariableOp,Technique_NN/dense_25/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_25/MatMul/ReadVariableOp+Technique_NN/dense_25/MatMul/ReadVariableOp2\
,Technique_NN/dense_26/BiasAdd/ReadVariableOp,Technique_NN/dense_26/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_26/MatMul/ReadVariableOp+Technique_NN/dense_26/MatMul/ReadVariableOp2\
,Technique_NN/dense_27/BiasAdd/ReadVariableOp,Technique_NN/dense_27/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_27/MatMul/ReadVariableOp+Technique_NN/dense_27/MatMul/ReadVariableOp2\
,Technique_NN/dense_28/BiasAdd/ReadVariableOp,Technique_NN/dense_28/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_28/MatMul/ReadVariableOp+Technique_NN/dense_28/MatMul/ReadVariableOp2\
,Technique_NN/dense_29/BiasAdd/ReadVariableOp,Technique_NN/dense_29/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_29/MatMul/ReadVariableOp+Technique_NN/dense_29/MatMul/ReadVariableOp2^
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
E__inference_dense_25_layer_call_and_return_conditional_losses_2339883

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
�(
�
E__inference_model1_2_layer_call_and_return_conditional_losses_2340337
input_group
input_technique#
group_nn_2340258:	� 
group_nn_2340260: "
group_nn_2340262:  
group_nn_2340264: "
group_nn_2340266:  
group_nn_2340268: "
group_nn_2340270:  
group_nn_2340272: "
group_nn_2340274:  
group_nn_2340276: "
group_nn_2340278: 
group_nn_2340280:'
technique_nn_2340283:	�@"
technique_nn_2340285:@&
technique_nn_2340287:@@"
technique_nn_2340289:@&
technique_nn_2340291:@@"
technique_nn_2340293:@&
technique_nn_2340295:@@"
technique_nn_2340297:@&
technique_nn_2340299:@@"
technique_nn_2340301:@&
technique_nn_2340303:@"
technique_nn_2340305:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinput_groupgroup_nn_2340258group_nn_2340260group_nn_2340262group_nn_2340264group_nn_2340266group_nn_2340268group_nn_2340270group_nn_2340272group_nn_2340274group_nn_2340276group_nn_2340278group_nn_2340280*
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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339661�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinput_techniquetechnique_nn_2340283technique_nn_2340285technique_nn_2340287technique_nn_2340289technique_nn_2340291technique_nn_2340293technique_nn_2340295technique_nn_2340297technique_nn_2340299technique_nn_2340301technique_nn_2340303technique_nn_2340305*
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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340045z
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
dot_2/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_2340334m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
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
�
�
*__inference_model1_2_layer_call_fn_2340656
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
GPU 2J 8� *N
fIRG
E__inference_model1_2_layer_call_and_return_conditional_losses_2340605o
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
E__inference_dense_20_layer_call_and_return_conditional_losses_2339499

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
�(
�
E__inference_model1_2_layer_call_and_return_conditional_losses_2340605

inputs
inputs_1#
group_nn_2340539:	� 
group_nn_2340541: "
group_nn_2340543:  
group_nn_2340545: "
group_nn_2340547:  
group_nn_2340549: "
group_nn_2340551:  
group_nn_2340553: "
group_nn_2340555:  
group_nn_2340557: "
group_nn_2340559: 
group_nn_2340561:'
technique_nn_2340564:	�@"
technique_nn_2340566:@&
technique_nn_2340568:@@"
technique_nn_2340570:@&
technique_nn_2340572:@@"
technique_nn_2340574:@&
technique_nn_2340576:@@"
technique_nn_2340578:@&
technique_nn_2340580:@@"
technique_nn_2340582:@&
technique_nn_2340584:@"
technique_nn_2340586:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinputsgroup_nn_2340539group_nn_2340541group_nn_2340543group_nn_2340545group_nn_2340547group_nn_2340549group_nn_2340551group_nn_2340553group_nn_2340555group_nn_2340557group_nn_2340559group_nn_2340561*
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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339724�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinputs_1technique_nn_2340564technique_nn_2340566technique_nn_2340568technique_nn_2340570technique_nn_2340572technique_nn_2340574technique_nn_2340576technique_nn_2340578technique_nn_2340580technique_nn_2340582technique_nn_2340584technique_nn_2340586*
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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340108z
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
dot_2/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_2340334m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
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
E__inference_dense_24_layer_call_and_return_conditional_losses_2341558

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

�
.__inference_Technique_NN_layer_call_fn_2341353

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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340108o
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
�
*__inference_Group_NN_layer_call_fn_2339688
dense_20_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339661o
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
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_20_input
�3
�	
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341397

inputs:
'dense_25_matmul_readvariableop_resource:	�@6
(dense_25_biasadd_readvariableop_resource:@9
'dense_26_matmul_readvariableop_resource:@@6
(dense_26_biasadd_readvariableop_resource:@9
'dense_27_matmul_readvariableop_resource:@@6
(dense_27_biasadd_readvariableop_resource:@9
'dense_28_matmul_readvariableop_resource:@@6
(dense_28_biasadd_readvariableop_resource:@9
'dense_29_matmul_readvariableop_resource:@@6
(dense_29_biasadd_readvariableop_resource:@:
(output_nn_matmul_readvariableop_resource:@7
)output_nn_biasadd_readvariableop_resource:
identity��dense_25/BiasAdd/ReadVariableOp�dense_25/MatMul/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0{
dense_25/MatMulMatMulinputs&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_26/MatMulMatMuldense_25/BiasAdd:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_28/MatMulMatMuldense_27/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
output_NN/MatMulMatMuldense_29/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_2339516

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

�
*__inference_Group_NN_layer_call_fn_2341178

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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339661o
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2339917

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
��
�.
#__inference__traced_restore_2342413
file_prefix3
 assignvariableop_dense_20_kernel:	� .
 assignvariableop_1_dense_20_bias: 4
"assignvariableop_2_dense_21_kernel:  .
 assignvariableop_3_dense_21_bias: 4
"assignvariableop_4_dense_22_kernel:  .
 assignvariableop_5_dense_22_bias: 4
"assignvariableop_6_dense_23_kernel:  .
 assignvariableop_7_dense_23_bias: 4
"assignvariableop_8_dense_24_kernel:  .
 assignvariableop_9_dense_24_bias: 8
&assignvariableop_10_output_nn_kernel_1: 2
$assignvariableop_11_output_nn_bias_1:6
#assignvariableop_12_dense_25_kernel:	�@/
!assignvariableop_13_dense_25_bias:@5
#assignvariableop_14_dense_26_kernel:@@/
!assignvariableop_15_dense_26_bias:@5
#assignvariableop_16_dense_27_kernel:@@/
!assignvariableop_17_dense_27_bias:@5
#assignvariableop_18_dense_28_kernel:@@/
!assignvariableop_19_dense_28_bias:@5
#assignvariableop_20_dense_29_kernel:@@/
!assignvariableop_21_dense_29_bias:@6
$assignvariableop_22_output_nn_kernel:@0
"assignvariableop_23_output_nn_bias:'
assignvariableop_24_iteration:	 +
!assignvariableop_25_learning_rate: =
*assignvariableop_26_adam_m_dense_20_kernel:	� =
*assignvariableop_27_adam_v_dense_20_kernel:	� 6
(assignvariableop_28_adam_m_dense_20_bias: 6
(assignvariableop_29_adam_v_dense_20_bias: <
*assignvariableop_30_adam_m_dense_21_kernel:  <
*assignvariableop_31_adam_v_dense_21_kernel:  6
(assignvariableop_32_adam_m_dense_21_bias: 6
(assignvariableop_33_adam_v_dense_21_bias: <
*assignvariableop_34_adam_m_dense_22_kernel:  <
*assignvariableop_35_adam_v_dense_22_kernel:  6
(assignvariableop_36_adam_m_dense_22_bias: 6
(assignvariableop_37_adam_v_dense_22_bias: <
*assignvariableop_38_adam_m_dense_23_kernel:  <
*assignvariableop_39_adam_v_dense_23_kernel:  6
(assignvariableop_40_adam_m_dense_23_bias: 6
(assignvariableop_41_adam_v_dense_23_bias: <
*assignvariableop_42_adam_m_dense_24_kernel:  <
*assignvariableop_43_adam_v_dense_24_kernel:  6
(assignvariableop_44_adam_m_dense_24_bias: 6
(assignvariableop_45_adam_v_dense_24_bias: ?
-assignvariableop_46_adam_m_output_nn_kernel_1: ?
-assignvariableop_47_adam_v_output_nn_kernel_1: 9
+assignvariableop_48_adam_m_output_nn_bias_1:9
+assignvariableop_49_adam_v_output_nn_bias_1:=
*assignvariableop_50_adam_m_dense_25_kernel:	�@=
*assignvariableop_51_adam_v_dense_25_kernel:	�@6
(assignvariableop_52_adam_m_dense_25_bias:@6
(assignvariableop_53_adam_v_dense_25_bias:@<
*assignvariableop_54_adam_m_dense_26_kernel:@@<
*assignvariableop_55_adam_v_dense_26_kernel:@@6
(assignvariableop_56_adam_m_dense_26_bias:@6
(assignvariableop_57_adam_v_dense_26_bias:@<
*assignvariableop_58_adam_m_dense_27_kernel:@@<
*assignvariableop_59_adam_v_dense_27_kernel:@@6
(assignvariableop_60_adam_m_dense_27_bias:@6
(assignvariableop_61_adam_v_dense_27_bias:@<
*assignvariableop_62_adam_m_dense_28_kernel:@@<
*assignvariableop_63_adam_v_dense_28_kernel:@@6
(assignvariableop_64_adam_m_dense_28_bias:@6
(assignvariableop_65_adam_v_dense_28_bias:@<
*assignvariableop_66_adam_m_dense_29_kernel:@@<
*assignvariableop_67_adam_v_dense_29_kernel:@@6
(assignvariableop_68_adam_m_dense_29_bias:@6
(assignvariableop_69_adam_v_dense_29_bias:@=
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
AssignVariableOpAssignVariableOp assignvariableop_dense_20_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_20_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_21_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_21_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_22_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_22_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_23_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_23_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_24_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_24_biasIdentity_9:output:0"/device:CPU:0*&
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
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_25_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_25_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_26_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_26_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_27_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_27_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_28_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_28_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_29_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_29_biasIdentity_21:output:0"/device:CPU:0*&
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
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_20_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_20_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_dense_20_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_dense_20_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_dense_21_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_dense_21_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_m_dense_21_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_v_dense_21_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_22_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_22_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_dense_22_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_dense_22_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_23_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_23_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_dense_23_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_dense_23_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_24_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_24_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_dense_24_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_dense_24_biasIdentity_45:output:0"/device:CPU:0*&
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
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_m_dense_25_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_v_dense_25_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_m_dense_25_biasIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_v_dense_25_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_m_dense_26_kernelIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_v_dense_26_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_m_dense_26_biasIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp(assignvariableop_57_adam_v_dense_26_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_m_dense_27_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_v_dense_27_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_m_dense_27_biasIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_v_dense_27_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_m_dense_28_kernelIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_v_dense_28_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_m_dense_28_biasIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp(assignvariableop_65_adam_v_dense_28_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_m_dense_29_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_v_dense_29_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_m_dense_29_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_v_dense_29_biasIdentity_69:output:0"/device:CPU:0*&
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
�

�
E__inference_dense_29_layer_call_and_return_conditional_losses_2341676

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
E__inference_dense_29_layer_call_and_return_conditional_losses_2339951

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
*__inference_dense_21_layer_call_fn_2341487

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
GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2339516o
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
�
�
*__inference_dense_29_layer_call_fn_2341665

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
GPU 2J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_2339951o
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2339900

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
n
B__inference_dot_2_layer_call_and_return_conditional_losses_2341459
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
�3
�	
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341251

inputs:
'dense_20_matmul_readvariableop_resource:	� 6
(dense_20_biasadd_readvariableop_resource: 9
'dense_21_matmul_readvariableop_resource:  6
(dense_21_biasadd_readvariableop_resource: 9
'dense_22_matmul_readvariableop_resource:  6
(dense_22_biasadd_readvariableop_resource: 9
'dense_23_matmul_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 9
'dense_24_matmul_readvariableop_resource:  6
(dense_24_biasadd_readvariableop_resource: :
(output_nn_matmul_readvariableop_resource: 7
)output_nn_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_21/MatMulMatMuldense_20/BiasAdd:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output_NN/MatMulMatMuldense_24/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
*__inference_Group_NN_layer_call_fn_2341207

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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339724o
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
�
�
*__inference_dense_28_layer_call_fn_2341645

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
GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2339934o
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
.__inference_Technique_NN_layer_call_fn_2340135
dense_25_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340108o
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
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_25_input
�

�
E__inference_dense_27_layer_call_and_return_conditional_losses_2341636

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
+__inference_output_NN_layer_call_fn_2341685

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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339967o
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
�

�
E__inference_dense_28_layer_call_and_return_conditional_losses_2339934

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
�(
�
E__inference_model1_2_layer_call_and_return_conditional_losses_2340481

inputs
inputs_1#
group_nn_2340415:	� 
group_nn_2340417: "
group_nn_2340419:  
group_nn_2340421: "
group_nn_2340423:  
group_nn_2340425: "
group_nn_2340427:  
group_nn_2340429: "
group_nn_2340431:  
group_nn_2340433: "
group_nn_2340435: 
group_nn_2340437:'
technique_nn_2340440:	�@"
technique_nn_2340442:@&
technique_nn_2340444:@@"
technique_nn_2340446:@&
technique_nn_2340448:@@"
technique_nn_2340450:@&
technique_nn_2340452:@@"
technique_nn_2340454:@&
technique_nn_2340456:@@"
technique_nn_2340458:@&
technique_nn_2340460:@"
technique_nn_2340462:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinputsgroup_nn_2340415group_nn_2340417group_nn_2340419group_nn_2340421group_nn_2340423group_nn_2340425group_nn_2340427group_nn_2340429group_nn_2340431group_nn_2340433group_nn_2340435group_nn_2340437*
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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339661�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinputs_1technique_nn_2340440technique_nn_2340442technique_nn_2340444technique_nn_2340446technique_nn_2340448technique_nn_2340450technique_nn_2340452technique_nn_2340454technique_nn_2340456technique_nn_2340458technique_nn_2340460technique_nn_2340462*
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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340045z
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
dot_2/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_2340334m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
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
E__inference_dense_22_layer_call_and_return_conditional_losses_2339533

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
F__inference_output_NN_layer_call_and_return_conditional_losses_2339967

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
F__inference_output_NN_layer_call_and_return_conditional_losses_2339583

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
l
B__inference_dot_2_layer_call_and_return_conditional_losses_2340334

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
.__inference_Technique_NN_layer_call_fn_2340072
dense_25_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340045o
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
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_25_input
�	
�
E__inference_dense_25_layer_call_and_return_conditional_losses_2341596

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
E__inference_dense_24_layer_call_and_return_conditional_losses_2339567

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
�3
�	
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341295

inputs:
'dense_20_matmul_readvariableop_resource:	� 6
(dense_20_biasadd_readvariableop_resource: 9
'dense_21_matmul_readvariableop_resource:  6
(dense_21_biasadd_readvariableop_resource: 9
'dense_22_matmul_readvariableop_resource:  6
(dense_22_biasadd_readvariableop_resource: 9
'dense_23_matmul_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 9
'dense_24_matmul_readvariableop_resource:  6
(dense_24_biasadd_readvariableop_resource: :
(output_nn_matmul_readvariableop_resource: 7
)output_nn_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_24/BiasAdd/ReadVariableOp�dense_24/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0{
dense_20/MatMulMatMulinputs&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_21/MatMulMatMuldense_20/BiasAdd:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_22/MatMulMatMuldense_21/Relu:activations:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output_NN/MatMulMatMuldense_24/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_21_layer_call_and_return_conditional_losses_2341498

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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339661

inputs#
dense_20_2339630:	� 
dense_20_2339632: "
dense_21_2339635:  
dense_21_2339637: "
dense_22_2339640:  
dense_22_2339642: "
dense_23_2339645:  
dense_23_2339647: "
dense_24_2339650:  
dense_24_2339652: #
output_nn_2339655: 
output_nn_2339657:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_2339630dense_20_2339632*
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
GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2339499�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_2339635dense_21_2339637*
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
GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2339516�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2339640dense_22_2339642*
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
GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2339533�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2339645dense_23_2339647*
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
GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2339550�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2339650dense_24_2339652*
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
GPU 2J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2339567�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0output_nn_2339655output_nn_2339657*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339583y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
E__inference_dense_20_layer_call_and_return_conditional_losses_2341478

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
�

�
E__inference_dense_23_layer_call_and_return_conditional_losses_2339550

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
�
*__inference_Group_NN_layer_call_fn_2339751
dense_20_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339724o
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
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_20_input
� 
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339724

inputs#
dense_20_2339693:	� 
dense_20_2339695: "
dense_21_2339698:  
dense_21_2339700: "
dense_22_2339703:  
dense_22_2339705: "
dense_23_2339708:  
dense_23_2339710: "
dense_24_2339713:  
dense_24_2339715: #
output_nn_2339718: 
output_nn_2339720:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinputsdense_20_2339693dense_20_2339695*
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
GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2339499�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_2339698dense_21_2339700*
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
GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2339516�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2339703dense_22_2339705*
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
GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2339533�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2339708dense_23_2339710*
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
GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2339550�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2339713dense_24_2339715*
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
GPU 2J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2339567�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0output_nn_2339718output_nn_2339720*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339583y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_dense_26_layer_call_fn_2341605

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
GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2339900o
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
�!
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2339974
dense_25_input#
dense_25_2339884:	�@
dense_25_2339886:@"
dense_26_2339901:@@
dense_26_2339903:@"
dense_27_2339918:@@
dense_27_2339920:@"
dense_28_2339935:@@
dense_28_2339937:@"
dense_29_2339952:@@
dense_29_2339954:@#
output_nn_2339968:@
output_nn_2339970:
identity�� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_25/StatefulPartitionedCallStatefulPartitionedCalldense_25_inputdense_25_2339884dense_25_2339886*
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
GPU 2J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2339883�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_2339901dense_26_2339903*
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
GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2339900�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2339918dense_27_2339920*
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
GPU 2J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2339917�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_2339935dense_28_2339937*
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
GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2339934�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2339952dense_29_2339954*
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
GPU 2J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_2339951�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0output_nn_2339968output_nn_2339970*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339967y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_25_input
� 
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340045

inputs#
dense_25_2340014:	�@
dense_25_2340016:@"
dense_26_2340019:@@
dense_26_2340021:@"
dense_27_2340024:@@
dense_27_2340026:@"
dense_28_2340029:@@
dense_28_2340031:@"
dense_29_2340034:@@
dense_29_2340036:@#
output_nn_2340039:@
output_nn_2340041:
identity�� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25_2340014dense_25_2340016*
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
GPU 2J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2339883�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_2340019dense_26_2340021*
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
GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2339900�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2340024dense_27_2340026*
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
GPU 2J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2339917�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_2340029dense_28_2340031*
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
GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2339934�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2340034dense_29_2340036*
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
GPU 2J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_2339951�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0output_nn_2340039output_nn_2340041*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339967y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_28_layer_call_and_return_conditional_losses_2341656

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
*__inference_dense_23_layer_call_fn_2341527

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
GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2339550o
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

�
E__inference_dense_22_layer_call_and_return_conditional_losses_2341518

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
�
�
%__inference_signature_wrapper_2340829
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
GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2339485o
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
�
�
*__inference_dense_22_layer_call_fn_2341507

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
GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2339533o
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

�
.__inference_Technique_NN_layer_call_fn_2341324

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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340045o
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
�(
�
E__inference_model1_2_layer_call_and_return_conditional_losses_2340407
input_group
input_technique#
group_nn_2340341:	� 
group_nn_2340343: "
group_nn_2340345:  
group_nn_2340347: "
group_nn_2340349:  
group_nn_2340351: "
group_nn_2340353:  
group_nn_2340355: "
group_nn_2340357:  
group_nn_2340359: "
group_nn_2340361: 
group_nn_2340363:'
technique_nn_2340366:	�@"
technique_nn_2340368:@&
technique_nn_2340370:@@"
technique_nn_2340372:@&
technique_nn_2340374:@@"
technique_nn_2340376:@&
technique_nn_2340378:@@"
technique_nn_2340380:@&
technique_nn_2340382:@@"
technique_nn_2340384:@&
technique_nn_2340386:@"
technique_nn_2340388:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinput_groupgroup_nn_2340341group_nn_2340343group_nn_2340345group_nn_2340347group_nn_2340349group_nn_2340351group_nn_2340353group_nn_2340355group_nn_2340357group_nn_2340359group_nn_2340361group_nn_2340363*
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
GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339724�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinput_techniquetechnique_nn_2340366technique_nn_2340368technique_nn_2340370technique_nn_2340372technique_nn_2340374technique_nn_2340376technique_nn_2340378technique_nn_2340380technique_nn_2340382technique_nn_2340384technique_nn_2340386technique_nn_2340388*
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
GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340108z
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
dot_2/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_2340334m
IdentityIdentitydot_2/PartitionedCall:output:0^NoOp*
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
*__inference_dense_25_layer_call_fn_2341586

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
GPU 2J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2339883o
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
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339624
dense_20_input#
dense_20_2339593:	� 
dense_20_2339595: "
dense_21_2339598:  
dense_21_2339600: "
dense_22_2339603:  
dense_22_2339605: "
dense_23_2339608:  
dense_23_2339610: "
dense_24_2339613:  
dense_24_2339615: #
output_nn_2339618: 
output_nn_2339620:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_2339593dense_20_2339595*
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
GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2339499�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_2339598dense_21_2339600*
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
GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2339516�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2339603dense_22_2339605*
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
GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2339533�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2339608dense_23_2339610*
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
GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2339550�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2339613dense_24_2339615*
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
GPU 2J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2339567�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0output_nn_2339618output_nn_2339620*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339583y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_20_input
�
�
*__inference_dense_24_layer_call_fn_2341547

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
GPU 2J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2339567o
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
�
�
*__inference_dense_27_layer_call_fn_2341625

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
GPU 2J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2339917o
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
�
�
*__inference_model1_2_layer_call_fn_2340532
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
GPU 2J 8� *N
fIRG
E__inference_model1_2_layer_call_and_return_conditional_losses_2340481o
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
�
�
*__inference_model1_2_layer_call_fn_2340883
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
GPU 2J 8� *N
fIRG
E__inference_model1_2_layer_call_and_return_conditional_losses_2340481o
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
��
�
"__inference__wrapped_model_2339485
input_group
input_techniqueL
9model1_2_group_nn_dense_20_matmul_readvariableop_resource:	� H
:model1_2_group_nn_dense_20_biasadd_readvariableop_resource: K
9model1_2_group_nn_dense_21_matmul_readvariableop_resource:  H
:model1_2_group_nn_dense_21_biasadd_readvariableop_resource: K
9model1_2_group_nn_dense_22_matmul_readvariableop_resource:  H
:model1_2_group_nn_dense_22_biasadd_readvariableop_resource: K
9model1_2_group_nn_dense_23_matmul_readvariableop_resource:  H
:model1_2_group_nn_dense_23_biasadd_readvariableop_resource: K
9model1_2_group_nn_dense_24_matmul_readvariableop_resource:  H
:model1_2_group_nn_dense_24_biasadd_readvariableop_resource: L
:model1_2_group_nn_output_nn_matmul_readvariableop_resource: I
;model1_2_group_nn_output_nn_biasadd_readvariableop_resource:P
=model1_2_technique_nn_dense_25_matmul_readvariableop_resource:	�@L
>model1_2_technique_nn_dense_25_biasadd_readvariableop_resource:@O
=model1_2_technique_nn_dense_26_matmul_readvariableop_resource:@@L
>model1_2_technique_nn_dense_26_biasadd_readvariableop_resource:@O
=model1_2_technique_nn_dense_27_matmul_readvariableop_resource:@@L
>model1_2_technique_nn_dense_27_biasadd_readvariableop_resource:@O
=model1_2_technique_nn_dense_28_matmul_readvariableop_resource:@@L
>model1_2_technique_nn_dense_28_biasadd_readvariableop_resource:@O
=model1_2_technique_nn_dense_29_matmul_readvariableop_resource:@@L
>model1_2_technique_nn_dense_29_biasadd_readvariableop_resource:@P
>model1_2_technique_nn_output_nn_matmul_readvariableop_resource:@M
?model1_2_technique_nn_output_nn_biasadd_readvariableop_resource:
identity��1model1_2/Group_NN/dense_20/BiasAdd/ReadVariableOp�0model1_2/Group_NN/dense_20/MatMul/ReadVariableOp�1model1_2/Group_NN/dense_21/BiasAdd/ReadVariableOp�0model1_2/Group_NN/dense_21/MatMul/ReadVariableOp�1model1_2/Group_NN/dense_22/BiasAdd/ReadVariableOp�0model1_2/Group_NN/dense_22/MatMul/ReadVariableOp�1model1_2/Group_NN/dense_23/BiasAdd/ReadVariableOp�0model1_2/Group_NN/dense_23/MatMul/ReadVariableOp�1model1_2/Group_NN/dense_24/BiasAdd/ReadVariableOp�0model1_2/Group_NN/dense_24/MatMul/ReadVariableOp�2model1_2/Group_NN/output_NN/BiasAdd/ReadVariableOp�1model1_2/Group_NN/output_NN/MatMul/ReadVariableOp�5model1_2/Technique_NN/dense_25/BiasAdd/ReadVariableOp�4model1_2/Technique_NN/dense_25/MatMul/ReadVariableOp�5model1_2/Technique_NN/dense_26/BiasAdd/ReadVariableOp�4model1_2/Technique_NN/dense_26/MatMul/ReadVariableOp�5model1_2/Technique_NN/dense_27/BiasAdd/ReadVariableOp�4model1_2/Technique_NN/dense_27/MatMul/ReadVariableOp�5model1_2/Technique_NN/dense_28/BiasAdd/ReadVariableOp�4model1_2/Technique_NN/dense_28/MatMul/ReadVariableOp�5model1_2/Technique_NN/dense_29/BiasAdd/ReadVariableOp�4model1_2/Technique_NN/dense_29/MatMul/ReadVariableOp�6model1_2/Technique_NN/output_NN/BiasAdd/ReadVariableOp�5model1_2/Technique_NN/output_NN/MatMul/ReadVariableOp�
0model1_2/Group_NN/dense_20/MatMul/ReadVariableOpReadVariableOp9model1_2_group_nn_dense_20_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
!model1_2/Group_NN/dense_20/MatMulMatMulinput_group8model1_2/Group_NN/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_2/Group_NN/dense_20/BiasAdd/ReadVariableOpReadVariableOp:model1_2_group_nn_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_2/Group_NN/dense_20/BiasAddBiasAdd+model1_2/Group_NN/dense_20/MatMul:product:09model1_2/Group_NN/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0model1_2/Group_NN/dense_21/MatMul/ReadVariableOpReadVariableOp9model1_2_group_nn_dense_21_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
!model1_2/Group_NN/dense_21/MatMulMatMul+model1_2/Group_NN/dense_20/BiasAdd:output:08model1_2/Group_NN/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_2/Group_NN/dense_21/BiasAdd/ReadVariableOpReadVariableOp:model1_2_group_nn_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_2/Group_NN/dense_21/BiasAddBiasAdd+model1_2/Group_NN/dense_21/MatMul:product:09model1_2/Group_NN/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1_2/Group_NN/dense_21/ReluRelu+model1_2/Group_NN/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
0model1_2/Group_NN/dense_22/MatMul/ReadVariableOpReadVariableOp9model1_2_group_nn_dense_22_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
!model1_2/Group_NN/dense_22/MatMulMatMul-model1_2/Group_NN/dense_21/Relu:activations:08model1_2/Group_NN/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_2/Group_NN/dense_22/BiasAdd/ReadVariableOpReadVariableOp:model1_2_group_nn_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_2/Group_NN/dense_22/BiasAddBiasAdd+model1_2/Group_NN/dense_22/MatMul:product:09model1_2/Group_NN/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1_2/Group_NN/dense_22/ReluRelu+model1_2/Group_NN/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
0model1_2/Group_NN/dense_23/MatMul/ReadVariableOpReadVariableOp9model1_2_group_nn_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
!model1_2/Group_NN/dense_23/MatMulMatMul-model1_2/Group_NN/dense_22/Relu:activations:08model1_2/Group_NN/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_2/Group_NN/dense_23/BiasAdd/ReadVariableOpReadVariableOp:model1_2_group_nn_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_2/Group_NN/dense_23/BiasAddBiasAdd+model1_2/Group_NN/dense_23/MatMul:product:09model1_2/Group_NN/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1_2/Group_NN/dense_23/ReluRelu+model1_2/Group_NN/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
0model1_2/Group_NN/dense_24/MatMul/ReadVariableOpReadVariableOp9model1_2_group_nn_dense_24_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
!model1_2/Group_NN/dense_24/MatMulMatMul-model1_2/Group_NN/dense_23/Relu:activations:08model1_2/Group_NN/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_2/Group_NN/dense_24/BiasAdd/ReadVariableOpReadVariableOp:model1_2_group_nn_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_2/Group_NN/dense_24/BiasAddBiasAdd+model1_2/Group_NN/dense_24/MatMul:product:09model1_2/Group_NN/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1_2/Group_NN/dense_24/ReluRelu+model1_2/Group_NN/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1model1_2/Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp:model1_2_group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
"model1_2/Group_NN/output_NN/MatMulMatMul-model1_2/Group_NN/dense_24/Relu:activations:09model1_2/Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2model1_2/Group_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp;model1_2_group_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model1_2/Group_NN/output_NN/BiasAddBiasAdd,model1_2/Group_NN/output_NN/MatMul:product:0:model1_2/Group_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4model1_2/Technique_NN/dense_25/MatMul/ReadVariableOpReadVariableOp=model1_2_technique_nn_dense_25_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
%model1_2/Technique_NN/dense_25/MatMulMatMulinput_technique<model1_2/Technique_NN/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_2/Technique_NN/dense_25/BiasAdd/ReadVariableOpReadVariableOp>model1_2_technique_nn_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_2/Technique_NN/dense_25/BiasAddBiasAdd/model1_2/Technique_NN/dense_25/MatMul:product:0=model1_2/Technique_NN/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
4model1_2/Technique_NN/dense_26/MatMul/ReadVariableOpReadVariableOp=model1_2_technique_nn_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
%model1_2/Technique_NN/dense_26/MatMulMatMul/model1_2/Technique_NN/dense_25/BiasAdd:output:0<model1_2/Technique_NN/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_2/Technique_NN/dense_26/BiasAdd/ReadVariableOpReadVariableOp>model1_2_technique_nn_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_2/Technique_NN/dense_26/BiasAddBiasAdd/model1_2/Technique_NN/dense_26/MatMul:product:0=model1_2/Technique_NN/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model1_2/Technique_NN/dense_26/ReluRelu/model1_2/Technique_NN/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
4model1_2/Technique_NN/dense_27/MatMul/ReadVariableOpReadVariableOp=model1_2_technique_nn_dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
%model1_2/Technique_NN/dense_27/MatMulMatMul1model1_2/Technique_NN/dense_26/Relu:activations:0<model1_2/Technique_NN/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_2/Technique_NN/dense_27/BiasAdd/ReadVariableOpReadVariableOp>model1_2_technique_nn_dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_2/Technique_NN/dense_27/BiasAddBiasAdd/model1_2/Technique_NN/dense_27/MatMul:product:0=model1_2/Technique_NN/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model1_2/Technique_NN/dense_27/ReluRelu/model1_2/Technique_NN/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
4model1_2/Technique_NN/dense_28/MatMul/ReadVariableOpReadVariableOp=model1_2_technique_nn_dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
%model1_2/Technique_NN/dense_28/MatMulMatMul1model1_2/Technique_NN/dense_27/Relu:activations:0<model1_2/Technique_NN/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_2/Technique_NN/dense_28/BiasAdd/ReadVariableOpReadVariableOp>model1_2_technique_nn_dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_2/Technique_NN/dense_28/BiasAddBiasAdd/model1_2/Technique_NN/dense_28/MatMul:product:0=model1_2/Technique_NN/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model1_2/Technique_NN/dense_28/ReluRelu/model1_2/Technique_NN/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
4model1_2/Technique_NN/dense_29/MatMul/ReadVariableOpReadVariableOp=model1_2_technique_nn_dense_29_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
%model1_2/Technique_NN/dense_29/MatMulMatMul1model1_2/Technique_NN/dense_28/Relu:activations:0<model1_2/Technique_NN/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_2/Technique_NN/dense_29/BiasAdd/ReadVariableOpReadVariableOp>model1_2_technique_nn_dense_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_2/Technique_NN/dense_29/BiasAddBiasAdd/model1_2/Technique_NN/dense_29/MatMul:product:0=model1_2/Technique_NN/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model1_2/Technique_NN/dense_29/ReluRelu/model1_2/Technique_NN/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
5model1_2/Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp>model1_2_technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model1_2/Technique_NN/output_NN/MatMulMatMul1model1_2/Technique_NN/dense_29/Relu:activations:0=model1_2/Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6model1_2/Technique_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp?model1_2_technique_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'model1_2/Technique_NN/output_NN/BiasAddBiasAdd0model1_2/Technique_NN/output_NN/MatMul:product:0>model1_2/Technique_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model1_2/l2_normalize/SquareSquare,model1_2/Group_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
+model1_2/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model1_2/l2_normalize/SumSum model1_2/l2_normalize/Square:y:04model1_2/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(d
model1_2/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
model1_2/l2_normalize/MaximumMaximum"model1_2/l2_normalize/Sum:output:0(model1_2/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������y
model1_2/l2_normalize/RsqrtRsqrt!model1_2/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
model1_2/l2_normalizeMul,model1_2/Group_NN/output_NN/BiasAdd:output:0model1_2/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
model1_2/l2_normalize_1/SquareSquare0model1_2/Technique_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
-model1_2/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model1_2/l2_normalize_1/SumSum"model1_2/l2_normalize_1/Square:y:06model1_2/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(f
!model1_2/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
model1_2/l2_normalize_1/MaximumMaximum$model1_2/l2_normalize_1/Sum:output:0*model1_2/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������}
model1_2/l2_normalize_1/RsqrtRsqrt#model1_2/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
model1_2/l2_normalize_1Mul0model1_2/Technique_NN/output_NN/BiasAdd:output:0!model1_2/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������_
model1_2/dot_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model1_2/dot_2/ExpandDims
ExpandDimsmodel1_2/l2_normalize:z:0&model1_2/dot_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������a
model1_2/dot_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model1_2/dot_2/ExpandDims_1
ExpandDimsmodel1_2/l2_normalize_1:z:0(model1_2/dot_2/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
model1_2/dot_2/MatMulBatchMatMulV2"model1_2/dot_2/ExpandDims:output:0$model1_2/dot_2/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������p
model1_2/dot_2/ShapeShapemodel1_2/dot_2/MatMul:output:0*
T0*
_output_shapes
::���
model1_2/dot_2/SqueezeSqueezemodel1_2/dot_2/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
n
IdentityIdentitymodel1_2/dot_2/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp2^model1_2/Group_NN/dense_20/BiasAdd/ReadVariableOp1^model1_2/Group_NN/dense_20/MatMul/ReadVariableOp2^model1_2/Group_NN/dense_21/BiasAdd/ReadVariableOp1^model1_2/Group_NN/dense_21/MatMul/ReadVariableOp2^model1_2/Group_NN/dense_22/BiasAdd/ReadVariableOp1^model1_2/Group_NN/dense_22/MatMul/ReadVariableOp2^model1_2/Group_NN/dense_23/BiasAdd/ReadVariableOp1^model1_2/Group_NN/dense_23/MatMul/ReadVariableOp2^model1_2/Group_NN/dense_24/BiasAdd/ReadVariableOp1^model1_2/Group_NN/dense_24/MatMul/ReadVariableOp3^model1_2/Group_NN/output_NN/BiasAdd/ReadVariableOp2^model1_2/Group_NN/output_NN/MatMul/ReadVariableOp6^model1_2/Technique_NN/dense_25/BiasAdd/ReadVariableOp5^model1_2/Technique_NN/dense_25/MatMul/ReadVariableOp6^model1_2/Technique_NN/dense_26/BiasAdd/ReadVariableOp5^model1_2/Technique_NN/dense_26/MatMul/ReadVariableOp6^model1_2/Technique_NN/dense_27/BiasAdd/ReadVariableOp5^model1_2/Technique_NN/dense_27/MatMul/ReadVariableOp6^model1_2/Technique_NN/dense_28/BiasAdd/ReadVariableOp5^model1_2/Technique_NN/dense_28/MatMul/ReadVariableOp6^model1_2/Technique_NN/dense_29/BiasAdd/ReadVariableOp5^model1_2/Technique_NN/dense_29/MatMul/ReadVariableOp7^model1_2/Technique_NN/output_NN/BiasAdd/ReadVariableOp6^model1_2/Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2f
1model1_2/Group_NN/dense_20/BiasAdd/ReadVariableOp1model1_2/Group_NN/dense_20/BiasAdd/ReadVariableOp2d
0model1_2/Group_NN/dense_20/MatMul/ReadVariableOp0model1_2/Group_NN/dense_20/MatMul/ReadVariableOp2f
1model1_2/Group_NN/dense_21/BiasAdd/ReadVariableOp1model1_2/Group_NN/dense_21/BiasAdd/ReadVariableOp2d
0model1_2/Group_NN/dense_21/MatMul/ReadVariableOp0model1_2/Group_NN/dense_21/MatMul/ReadVariableOp2f
1model1_2/Group_NN/dense_22/BiasAdd/ReadVariableOp1model1_2/Group_NN/dense_22/BiasAdd/ReadVariableOp2d
0model1_2/Group_NN/dense_22/MatMul/ReadVariableOp0model1_2/Group_NN/dense_22/MatMul/ReadVariableOp2f
1model1_2/Group_NN/dense_23/BiasAdd/ReadVariableOp1model1_2/Group_NN/dense_23/BiasAdd/ReadVariableOp2d
0model1_2/Group_NN/dense_23/MatMul/ReadVariableOp0model1_2/Group_NN/dense_23/MatMul/ReadVariableOp2f
1model1_2/Group_NN/dense_24/BiasAdd/ReadVariableOp1model1_2/Group_NN/dense_24/BiasAdd/ReadVariableOp2d
0model1_2/Group_NN/dense_24/MatMul/ReadVariableOp0model1_2/Group_NN/dense_24/MatMul/ReadVariableOp2h
2model1_2/Group_NN/output_NN/BiasAdd/ReadVariableOp2model1_2/Group_NN/output_NN/BiasAdd/ReadVariableOp2f
1model1_2/Group_NN/output_NN/MatMul/ReadVariableOp1model1_2/Group_NN/output_NN/MatMul/ReadVariableOp2n
5model1_2/Technique_NN/dense_25/BiasAdd/ReadVariableOp5model1_2/Technique_NN/dense_25/BiasAdd/ReadVariableOp2l
4model1_2/Technique_NN/dense_25/MatMul/ReadVariableOp4model1_2/Technique_NN/dense_25/MatMul/ReadVariableOp2n
5model1_2/Technique_NN/dense_26/BiasAdd/ReadVariableOp5model1_2/Technique_NN/dense_26/BiasAdd/ReadVariableOp2l
4model1_2/Technique_NN/dense_26/MatMul/ReadVariableOp4model1_2/Technique_NN/dense_26/MatMul/ReadVariableOp2n
5model1_2/Technique_NN/dense_27/BiasAdd/ReadVariableOp5model1_2/Technique_NN/dense_27/BiasAdd/ReadVariableOp2l
4model1_2/Technique_NN/dense_27/MatMul/ReadVariableOp4model1_2/Technique_NN/dense_27/MatMul/ReadVariableOp2n
5model1_2/Technique_NN/dense_28/BiasAdd/ReadVariableOp5model1_2/Technique_NN/dense_28/BiasAdd/ReadVariableOp2l
4model1_2/Technique_NN/dense_28/MatMul/ReadVariableOp4model1_2/Technique_NN/dense_28/MatMul/ReadVariableOp2n
5model1_2/Technique_NN/dense_29/BiasAdd/ReadVariableOp5model1_2/Technique_NN/dense_29/BiasAdd/ReadVariableOp2l
4model1_2/Technique_NN/dense_29/MatMul/ReadVariableOp4model1_2/Technique_NN/dense_29/MatMul/ReadVariableOp2p
6model1_2/Technique_NN/output_NN/BiasAdd/ReadVariableOp6model1_2/Technique_NN/output_NN/BiasAdd/ReadVariableOp2n
5model1_2/Technique_NN/output_NN/MatMul/ReadVariableOp5model1_2/Technique_NN/output_NN/MatMul/ReadVariableOp:YU
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2341577

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
�
S
'__inference_dot_2_layer_call_fn_2341447
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
GPU 2J 8� *K
fFRD
B__inference_dot_2_layer_call_and_return_conditional_losses_2340334`
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
�
F__inference_output_NN_layer_call_and_return_conditional_losses_2341695

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
�
�
E__inference_model1_2_layer_call_and_return_conditional_losses_2341149
inputs_input_group
inputs_input_techniqueC
0group_nn_dense_20_matmul_readvariableop_resource:	� ?
1group_nn_dense_20_biasadd_readvariableop_resource: B
0group_nn_dense_21_matmul_readvariableop_resource:  ?
1group_nn_dense_21_biasadd_readvariableop_resource: B
0group_nn_dense_22_matmul_readvariableop_resource:  ?
1group_nn_dense_22_biasadd_readvariableop_resource: B
0group_nn_dense_23_matmul_readvariableop_resource:  ?
1group_nn_dense_23_biasadd_readvariableop_resource: B
0group_nn_dense_24_matmul_readvariableop_resource:  ?
1group_nn_dense_24_biasadd_readvariableop_resource: C
1group_nn_output_nn_matmul_readvariableop_resource: @
2group_nn_output_nn_biasadd_readvariableop_resource:G
4technique_nn_dense_25_matmul_readvariableop_resource:	�@C
5technique_nn_dense_25_biasadd_readvariableop_resource:@F
4technique_nn_dense_26_matmul_readvariableop_resource:@@C
5technique_nn_dense_26_biasadd_readvariableop_resource:@F
4technique_nn_dense_27_matmul_readvariableop_resource:@@C
5technique_nn_dense_27_biasadd_readvariableop_resource:@F
4technique_nn_dense_28_matmul_readvariableop_resource:@@C
5technique_nn_dense_28_biasadd_readvariableop_resource:@F
4technique_nn_dense_29_matmul_readvariableop_resource:@@C
5technique_nn_dense_29_biasadd_readvariableop_resource:@G
5technique_nn_output_nn_matmul_readvariableop_resource:@D
6technique_nn_output_nn_biasadd_readvariableop_resource:
identity��(Group_NN/dense_20/BiasAdd/ReadVariableOp�'Group_NN/dense_20/MatMul/ReadVariableOp�(Group_NN/dense_21/BiasAdd/ReadVariableOp�'Group_NN/dense_21/MatMul/ReadVariableOp�(Group_NN/dense_22/BiasAdd/ReadVariableOp�'Group_NN/dense_22/MatMul/ReadVariableOp�(Group_NN/dense_23/BiasAdd/ReadVariableOp�'Group_NN/dense_23/MatMul/ReadVariableOp�(Group_NN/dense_24/BiasAdd/ReadVariableOp�'Group_NN/dense_24/MatMul/ReadVariableOp�)Group_NN/output_NN/BiasAdd/ReadVariableOp�(Group_NN/output_NN/MatMul/ReadVariableOp�,Technique_NN/dense_25/BiasAdd/ReadVariableOp�+Technique_NN/dense_25/MatMul/ReadVariableOp�,Technique_NN/dense_26/BiasAdd/ReadVariableOp�+Technique_NN/dense_26/MatMul/ReadVariableOp�,Technique_NN/dense_27/BiasAdd/ReadVariableOp�+Technique_NN/dense_27/MatMul/ReadVariableOp�,Technique_NN/dense_28/BiasAdd/ReadVariableOp�+Technique_NN/dense_28/MatMul/ReadVariableOp�,Technique_NN/dense_29/BiasAdd/ReadVariableOp�+Technique_NN/dense_29/MatMul/ReadVariableOp�-Technique_NN/output_NN/BiasAdd/ReadVariableOp�,Technique_NN/output_NN/MatMul/ReadVariableOp�
'Group_NN/dense_20/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_20_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
Group_NN/dense_20/MatMulMatMulinputs_input_group/Group_NN/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_20/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_20/BiasAddBiasAdd"Group_NN/dense_20/MatMul:product:00Group_NN/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_21/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_21_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_21/MatMulMatMul"Group_NN/dense_20/BiasAdd:output:0/Group_NN/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_21/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_21/BiasAddBiasAdd"Group_NN/dense_21/MatMul:product:00Group_NN/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_21/ReluRelu"Group_NN/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_22/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_22_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_22/MatMulMatMul$Group_NN/dense_21/Relu:activations:0/Group_NN/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_22/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_22/BiasAddBiasAdd"Group_NN/dense_22/MatMul:product:00Group_NN/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_22/ReluRelu"Group_NN/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_23/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_23/MatMulMatMul$Group_NN/dense_22/Relu:activations:0/Group_NN/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_23/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_23/BiasAddBiasAdd"Group_NN/dense_23/MatMul:product:00Group_NN/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_23/ReluRelu"Group_NN/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_24/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_24_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_24/MatMulMatMul$Group_NN/dense_23/Relu:activations:0/Group_NN/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_24/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_24/BiasAddBiasAdd"Group_NN/dense_24/MatMul:product:00Group_NN/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_24/ReluRelu"Group_NN/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp1group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Group_NN/output_NN/MatMulMatMul$Group_NN/dense_24/Relu:activations:00Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
+Technique_NN/dense_25/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_25_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Technique_NN/dense_25/MatMulMatMulinputs_input_technique3Technique_NN/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_25/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_25/BiasAddBiasAdd&Technique_NN/dense_25/MatMul:product:04Technique_NN/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_26/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_26/MatMulMatMul&Technique_NN/dense_25/BiasAdd:output:03Technique_NN/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_26/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_26/BiasAddBiasAdd&Technique_NN/dense_26/MatMul:product:04Technique_NN/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_26/ReluRelu&Technique_NN/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_27/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_27_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_27/MatMulMatMul(Technique_NN/dense_26/Relu:activations:03Technique_NN/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_27/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_27_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_27/BiasAddBiasAdd&Technique_NN/dense_27/MatMul:product:04Technique_NN/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_27/ReluRelu&Technique_NN/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_28/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_28_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_28/MatMulMatMul(Technique_NN/dense_27/Relu:activations:03Technique_NN/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_28/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_28/BiasAddBiasAdd&Technique_NN/dense_28/MatMul:product:04Technique_NN/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_28/ReluRelu&Technique_NN/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_29/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_29_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_29/MatMulMatMul(Technique_NN/dense_28/Relu:activations:03Technique_NN/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_29/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_29/BiasAddBiasAdd&Technique_NN/dense_29/MatMul:product:04Technique_NN/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_29/ReluRelu&Technique_NN/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp5technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
Technique_NN/output_NN/MatMulMatMul(Technique_NN/dense_29/Relu:activations:04Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
:���������V
dot_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims
ExpandDimsl2_normalize:z:0dot_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������X
dot_2/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_2/ExpandDims_1
ExpandDimsl2_normalize_1:z:0dot_2/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
dot_2/MatMulBatchMatMulV2dot_2/ExpandDims:output:0dot_2/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������^
dot_2/ShapeShapedot_2/MatMul:output:0*
T0*
_output_shapes
::��x
dot_2/SqueezeSqueezedot_2/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
e
IdentityIdentitydot_2/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^Group_NN/dense_20/BiasAdd/ReadVariableOp(^Group_NN/dense_20/MatMul/ReadVariableOp)^Group_NN/dense_21/BiasAdd/ReadVariableOp(^Group_NN/dense_21/MatMul/ReadVariableOp)^Group_NN/dense_22/BiasAdd/ReadVariableOp(^Group_NN/dense_22/MatMul/ReadVariableOp)^Group_NN/dense_23/BiasAdd/ReadVariableOp(^Group_NN/dense_23/MatMul/ReadVariableOp)^Group_NN/dense_24/BiasAdd/ReadVariableOp(^Group_NN/dense_24/MatMul/ReadVariableOp*^Group_NN/output_NN/BiasAdd/ReadVariableOp)^Group_NN/output_NN/MatMul/ReadVariableOp-^Technique_NN/dense_25/BiasAdd/ReadVariableOp,^Technique_NN/dense_25/MatMul/ReadVariableOp-^Technique_NN/dense_26/BiasAdd/ReadVariableOp,^Technique_NN/dense_26/MatMul/ReadVariableOp-^Technique_NN/dense_27/BiasAdd/ReadVariableOp,^Technique_NN/dense_27/MatMul/ReadVariableOp-^Technique_NN/dense_28/BiasAdd/ReadVariableOp,^Technique_NN/dense_28/MatMul/ReadVariableOp-^Technique_NN/dense_29/BiasAdd/ReadVariableOp,^Technique_NN/dense_29/MatMul/ReadVariableOp.^Technique_NN/output_NN/BiasAdd/ReadVariableOp-^Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*k
_input_shapesZ
X:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : 2T
(Group_NN/dense_20/BiasAdd/ReadVariableOp(Group_NN/dense_20/BiasAdd/ReadVariableOp2R
'Group_NN/dense_20/MatMul/ReadVariableOp'Group_NN/dense_20/MatMul/ReadVariableOp2T
(Group_NN/dense_21/BiasAdd/ReadVariableOp(Group_NN/dense_21/BiasAdd/ReadVariableOp2R
'Group_NN/dense_21/MatMul/ReadVariableOp'Group_NN/dense_21/MatMul/ReadVariableOp2T
(Group_NN/dense_22/BiasAdd/ReadVariableOp(Group_NN/dense_22/BiasAdd/ReadVariableOp2R
'Group_NN/dense_22/MatMul/ReadVariableOp'Group_NN/dense_22/MatMul/ReadVariableOp2T
(Group_NN/dense_23/BiasAdd/ReadVariableOp(Group_NN/dense_23/BiasAdd/ReadVariableOp2R
'Group_NN/dense_23/MatMul/ReadVariableOp'Group_NN/dense_23/MatMul/ReadVariableOp2T
(Group_NN/dense_24/BiasAdd/ReadVariableOp(Group_NN/dense_24/BiasAdd/ReadVariableOp2R
'Group_NN/dense_24/MatMul/ReadVariableOp'Group_NN/dense_24/MatMul/ReadVariableOp2V
)Group_NN/output_NN/BiasAdd/ReadVariableOp)Group_NN/output_NN/BiasAdd/ReadVariableOp2T
(Group_NN/output_NN/MatMul/ReadVariableOp(Group_NN/output_NN/MatMul/ReadVariableOp2\
,Technique_NN/dense_25/BiasAdd/ReadVariableOp,Technique_NN/dense_25/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_25/MatMul/ReadVariableOp+Technique_NN/dense_25/MatMul/ReadVariableOp2\
,Technique_NN/dense_26/BiasAdd/ReadVariableOp,Technique_NN/dense_26/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_26/MatMul/ReadVariableOp+Technique_NN/dense_26/MatMul/ReadVariableOp2\
,Technique_NN/dense_27/BiasAdd/ReadVariableOp,Technique_NN/dense_27/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_27/MatMul/ReadVariableOp+Technique_NN/dense_27/MatMul/ReadVariableOp2\
,Technique_NN/dense_28/BiasAdd/ReadVariableOp,Technique_NN/dense_28/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_28/MatMul/ReadVariableOp+Technique_NN/dense_28/MatMul/ReadVariableOp2\
,Technique_NN/dense_29/BiasAdd/ReadVariableOp,Technique_NN/dense_29/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_29/MatMul/ReadVariableOp+Technique_NN/dense_29/MatMul/ReadVariableOp2^
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
�!
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340008
dense_25_input#
dense_25_2339977:	�@
dense_25_2339979:@"
dense_26_2339982:@@
dense_26_2339984:@"
dense_27_2339987:@@
dense_27_2339989:@"
dense_28_2339992:@@
dense_28_2339994:@"
dense_29_2339997:@@
dense_29_2339999:@#
output_nn_2340002:@
output_nn_2340004:
identity�� dense_25/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_25/StatefulPartitionedCallStatefulPartitionedCalldense_25_inputdense_25_2339977dense_25_2339979*
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
GPU 2J 8� *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_2339883�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_2339982dense_26_2339984*
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
GPU 2J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_2339900�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2339987dense_27_2339989*
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
GPU 2J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_2339917�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_27/StatefulPartitionedCall:output:0dense_28_2339992dense_28_2339994*
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
GPU 2J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_2339934�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_2339997dense_29_2339999*
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
GPU 2J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_2339951�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_29/StatefulPartitionedCall:output:0output_nn_2340002output_nn_2340004*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339967y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_25_input
ܔ
�D
 __inference__traced_save_2342175
file_prefix9
&read_disablecopyonread_dense_20_kernel:	� 4
&read_1_disablecopyonread_dense_20_bias: :
(read_2_disablecopyonread_dense_21_kernel:  4
&read_3_disablecopyonread_dense_21_bias: :
(read_4_disablecopyonread_dense_22_kernel:  4
&read_5_disablecopyonread_dense_22_bias: :
(read_6_disablecopyonread_dense_23_kernel:  4
&read_7_disablecopyonread_dense_23_bias: :
(read_8_disablecopyonread_dense_24_kernel:  4
&read_9_disablecopyonread_dense_24_bias: >
,read_10_disablecopyonread_output_nn_kernel_1: 8
*read_11_disablecopyonread_output_nn_bias_1:<
)read_12_disablecopyonread_dense_25_kernel:	�@5
'read_13_disablecopyonread_dense_25_bias:@;
)read_14_disablecopyonread_dense_26_kernel:@@5
'read_15_disablecopyonread_dense_26_bias:@;
)read_16_disablecopyonread_dense_27_kernel:@@5
'read_17_disablecopyonread_dense_27_bias:@;
)read_18_disablecopyonread_dense_28_kernel:@@5
'read_19_disablecopyonread_dense_28_bias:@;
)read_20_disablecopyonread_dense_29_kernel:@@5
'read_21_disablecopyonread_dense_29_bias:@<
*read_22_disablecopyonread_output_nn_kernel:@6
(read_23_disablecopyonread_output_nn_bias:-
#read_24_disablecopyonread_iteration:	 1
'read_25_disablecopyonread_learning_rate: C
0read_26_disablecopyonread_adam_m_dense_20_kernel:	� C
0read_27_disablecopyonread_adam_v_dense_20_kernel:	� <
.read_28_disablecopyonread_adam_m_dense_20_bias: <
.read_29_disablecopyonread_adam_v_dense_20_bias: B
0read_30_disablecopyonread_adam_m_dense_21_kernel:  B
0read_31_disablecopyonread_adam_v_dense_21_kernel:  <
.read_32_disablecopyonread_adam_m_dense_21_bias: <
.read_33_disablecopyonread_adam_v_dense_21_bias: B
0read_34_disablecopyonread_adam_m_dense_22_kernel:  B
0read_35_disablecopyonread_adam_v_dense_22_kernel:  <
.read_36_disablecopyonread_adam_m_dense_22_bias: <
.read_37_disablecopyonread_adam_v_dense_22_bias: B
0read_38_disablecopyonread_adam_m_dense_23_kernel:  B
0read_39_disablecopyonread_adam_v_dense_23_kernel:  <
.read_40_disablecopyonread_adam_m_dense_23_bias: <
.read_41_disablecopyonread_adam_v_dense_23_bias: B
0read_42_disablecopyonread_adam_m_dense_24_kernel:  B
0read_43_disablecopyonread_adam_v_dense_24_kernel:  <
.read_44_disablecopyonread_adam_m_dense_24_bias: <
.read_45_disablecopyonread_adam_v_dense_24_bias: E
3read_46_disablecopyonread_adam_m_output_nn_kernel_1: E
3read_47_disablecopyonread_adam_v_output_nn_kernel_1: ?
1read_48_disablecopyonread_adam_m_output_nn_bias_1:?
1read_49_disablecopyonread_adam_v_output_nn_bias_1:C
0read_50_disablecopyonread_adam_m_dense_25_kernel:	�@C
0read_51_disablecopyonread_adam_v_dense_25_kernel:	�@<
.read_52_disablecopyonread_adam_m_dense_25_bias:@<
.read_53_disablecopyonread_adam_v_dense_25_bias:@B
0read_54_disablecopyonread_adam_m_dense_26_kernel:@@B
0read_55_disablecopyonread_adam_v_dense_26_kernel:@@<
.read_56_disablecopyonread_adam_m_dense_26_bias:@<
.read_57_disablecopyonread_adam_v_dense_26_bias:@B
0read_58_disablecopyonread_adam_m_dense_27_kernel:@@B
0read_59_disablecopyonread_adam_v_dense_27_kernel:@@<
.read_60_disablecopyonread_adam_m_dense_27_bias:@<
.read_61_disablecopyonread_adam_v_dense_27_bias:@B
0read_62_disablecopyonread_adam_m_dense_28_kernel:@@B
0read_63_disablecopyonread_adam_v_dense_28_kernel:@@<
.read_64_disablecopyonread_adam_m_dense_28_bias:@<
.read_65_disablecopyonread_adam_v_dense_28_bias:@B
0read_66_disablecopyonread_adam_m_dense_29_kernel:@@B
0read_67_disablecopyonread_adam_v_dense_29_kernel:@@<
.read_68_disablecopyonread_adam_m_dense_29_bias:@<
.read_69_disablecopyonread_adam_v_dense_29_bias:@C
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
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_20_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
:	� z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_20_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_21_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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

:  z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_21_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_22_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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

:  z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_22_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_dense_23_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
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

:  z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_dense_23_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
: |
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_dense_24_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
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

:  z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_dense_24_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
:~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_dense_25_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_dense_25_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
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
:	�@|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_dense_25_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_dense_25_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_dense_26_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_dense_26_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
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

:@@|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_dense_26_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_dense_26_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_dense_27_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_dense_27_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
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

:@@|
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_dense_27_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_dense_27_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_dense_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_dense_28_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
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

:@@|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_dense_28_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_dense_28_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:@~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_dense_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_dense_29_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
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

:@@|
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_dense_29_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_dense_29_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
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
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_20_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
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
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_20_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
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
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_dense_20_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
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
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_dense_20_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
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
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_dense_21_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
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
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_dense_21_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_dense_21_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
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
Read_32/DisableCopyOnReadDisableCopyOnRead.read_32_disablecopyonread_adam_m_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp.read_32_disablecopyonread_adam_m_dense_21_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
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
Read_33/DisableCopyOnReadDisableCopyOnRead.read_33_disablecopyonread_adam_v_dense_21_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp.read_33_disablecopyonread_adam_v_dense_21_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
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
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_dense_22_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
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
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_dense_22_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
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
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_dense_22_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
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
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_dense_22_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
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
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_dense_23_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
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
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_dense_23_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
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
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_dense_23_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
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
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_dense_23_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
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
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_dense_24_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
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
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_dense_24_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_dense_24_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
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
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_m_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_m_dense_24_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
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
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_v_dense_24_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_v_dense_24_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
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
Read_50/DisableCopyOnReadDisableCopyOnRead0read_50_disablecopyonread_adam_m_dense_25_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp0read_50_disablecopyonread_adam_m_dense_25_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
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
Read_51/DisableCopyOnReadDisableCopyOnRead0read_51_disablecopyonread_adam_v_dense_25_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp0read_51_disablecopyonread_adam_v_dense_25_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*
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
Read_52/DisableCopyOnReadDisableCopyOnRead.read_52_disablecopyonread_adam_m_dense_25_bias"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp.read_52_disablecopyonread_adam_m_dense_25_bias^Read_52/DisableCopyOnRead"/device:CPU:0*
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
Read_53/DisableCopyOnReadDisableCopyOnRead.read_53_disablecopyonread_adam_v_dense_25_bias"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp.read_53_disablecopyonread_adam_v_dense_25_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
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
Read_54/DisableCopyOnReadDisableCopyOnRead0read_54_disablecopyonread_adam_m_dense_26_kernel"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp0read_54_disablecopyonread_adam_m_dense_26_kernel^Read_54/DisableCopyOnRead"/device:CPU:0*
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
Read_55/DisableCopyOnReadDisableCopyOnRead0read_55_disablecopyonread_adam_v_dense_26_kernel"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp0read_55_disablecopyonread_adam_v_dense_26_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*
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
Read_56/DisableCopyOnReadDisableCopyOnRead.read_56_disablecopyonread_adam_m_dense_26_bias"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp.read_56_disablecopyonread_adam_m_dense_26_bias^Read_56/DisableCopyOnRead"/device:CPU:0*
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
Read_57/DisableCopyOnReadDisableCopyOnRead.read_57_disablecopyonread_adam_v_dense_26_bias"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp.read_57_disablecopyonread_adam_v_dense_26_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
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
Read_58/DisableCopyOnReadDisableCopyOnRead0read_58_disablecopyonread_adam_m_dense_27_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp0read_58_disablecopyonread_adam_m_dense_27_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*
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
Read_59/DisableCopyOnReadDisableCopyOnRead0read_59_disablecopyonread_adam_v_dense_27_kernel"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp0read_59_disablecopyonread_adam_v_dense_27_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
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
Read_60/DisableCopyOnReadDisableCopyOnRead.read_60_disablecopyonread_adam_m_dense_27_bias"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp.read_60_disablecopyonread_adam_m_dense_27_bias^Read_60/DisableCopyOnRead"/device:CPU:0*
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
Read_61/DisableCopyOnReadDisableCopyOnRead.read_61_disablecopyonread_adam_v_dense_27_bias"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp.read_61_disablecopyonread_adam_v_dense_27_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
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
Read_62/DisableCopyOnReadDisableCopyOnRead0read_62_disablecopyonread_adam_m_dense_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp0read_62_disablecopyonread_adam_m_dense_28_kernel^Read_62/DisableCopyOnRead"/device:CPU:0*
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
Read_63/DisableCopyOnReadDisableCopyOnRead0read_63_disablecopyonread_adam_v_dense_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp0read_63_disablecopyonread_adam_v_dense_28_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
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
Read_64/DisableCopyOnReadDisableCopyOnRead.read_64_disablecopyonread_adam_m_dense_28_bias"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp.read_64_disablecopyonread_adam_m_dense_28_bias^Read_64/DisableCopyOnRead"/device:CPU:0*
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
Read_65/DisableCopyOnReadDisableCopyOnRead.read_65_disablecopyonread_adam_v_dense_28_bias"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp.read_65_disablecopyonread_adam_v_dense_28_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
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
Read_66/DisableCopyOnReadDisableCopyOnRead0read_66_disablecopyonread_adam_m_dense_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp0read_66_disablecopyonread_adam_m_dense_29_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*
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
Read_67/DisableCopyOnReadDisableCopyOnRead0read_67_disablecopyonread_adam_v_dense_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp0read_67_disablecopyonread_adam_v_dense_29_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*
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
Read_68/DisableCopyOnReadDisableCopyOnRead.read_68_disablecopyonread_adam_m_dense_29_bias"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp.read_68_disablecopyonread_adam_m_dense_29_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
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
Read_69/DisableCopyOnReadDisableCopyOnRead.read_69_disablecopyonread_adam_v_dense_29_bias"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp.read_69_disablecopyonread_adam_v_dense_29_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
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
�

�
E__inference_dense_26_layer_call_and_return_conditional_losses_2341616

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
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339590
dense_20_input#
dense_20_2339500:	� 
dense_20_2339502: "
dense_21_2339517:  
dense_21_2339519: "
dense_22_2339534:  
dense_22_2339536: "
dense_23_2339551:  
dense_23_2339553: "
dense_24_2339568:  
dense_24_2339570: #
output_nn_2339584: 
output_nn_2339586:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall� dense_24/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCalldense_20_inputdense_20_2339500dense_20_2339502*
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
GPU 2J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2339499�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_2339517dense_21_2339519*
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
GPU 2J 8� *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2339516�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_2339534dense_22_2339536*
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
GPU 2J 8� *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2339533�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2339551dense_23_2339553*
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
GPU 2J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2339550�
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_2339568dense_24_2339570*
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
GPU 2J 8� *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_2339567�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0output_nn_2339584output_nn_2339586*
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
GPU 2J 8� *O
fJRH
F__inference_output_NN_layer_call_and_return_conditional_losses_2339583y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:����������: : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_20_input"�
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
*__inference_model1_2_layer_call_fn_2340532
*__inference_model1_2_layer_call_fn_2340656
*__inference_model1_2_layer_call_fn_2340883
*__inference_model1_2_layer_call_fn_2340937�
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2340337
E__inference_model1_2_layer_call_and_return_conditional_losses_2340407
E__inference_model1_2_layer_call_and_return_conditional_losses_2341043
E__inference_model1_2_layer_call_and_return_conditional_losses_2341149�
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
"__inference__wrapped_model_2339485input_Groupinput_Technique"�
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
": 	� 2dense_20/kernel
: 2dense_20/bias
!:  2dense_21/kernel
: 2dense_21/bias
!:  2dense_22/kernel
: 2dense_22/bias
!:  2dense_23/kernel
: 2dense_23/bias
!:  2dense_24/kernel
: 2dense_24/bias
":  2output_NN/kernel
:2output_NN/bias
": 	�@2dense_25/kernel
:@2dense_25/bias
!:@@2dense_26/kernel
:@2dense_26/bias
!:@@2dense_27/kernel
:@2dense_27/bias
!:@@2dense_28/kernel
:@2dense_28/bias
!:@@2dense_29/kernel
:@2dense_29/bias
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
*__inference_model1_2_layer_call_fn_2340532input_Groupinput_Technique"�
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
*__inference_model1_2_layer_call_fn_2340656input_Groupinput_Technique"�
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
*__inference_model1_2_layer_call_fn_2340883inputs_input_groupinputs_input_technique"�
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
*__inference_model1_2_layer_call_fn_2340937inputs_input_groupinputs_input_technique"�
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2340337input_Groupinput_Technique"�
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2340407input_Groupinput_Technique"�
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2341043inputs_input_groupinputs_input_technique"�
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2341149inputs_input_groupinputs_input_technique"�
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
*__inference_Group_NN_layer_call_fn_2339688
*__inference_Group_NN_layer_call_fn_2339751
*__inference_Group_NN_layer_call_fn_2341178
*__inference_Group_NN_layer_call_fn_2341207�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339590
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339624
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341251
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341295�
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
.__inference_Technique_NN_layer_call_fn_2340072
.__inference_Technique_NN_layer_call_fn_2340135
.__inference_Technique_NN_layer_call_fn_2341324
.__inference_Technique_NN_layer_call_fn_2341353�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2339974
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340008
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341397
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341441�
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
'__inference_dot_2_layer_call_fn_2341447�
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
B__inference_dot_2_layer_call_and_return_conditional_losses_2341459�
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
%__inference_signature_wrapper_2340829input_Groupinput_Technique"�
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
*__inference_dense_20_layer_call_fn_2341468�
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
�
�trace_02�
E__inference_dense_20_layer_call_and_return_conditional_losses_2341478�
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
*__inference_dense_21_layer_call_fn_2341487�
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
E__inference_dense_21_layer_call_and_return_conditional_losses_2341498�
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
*__inference_dense_22_layer_call_fn_2341507�
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
E__inference_dense_22_layer_call_and_return_conditional_losses_2341518�
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
*__inference_dense_23_layer_call_fn_2341527�
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
E__inference_dense_23_layer_call_and_return_conditional_losses_2341538�
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
*__inference_dense_24_layer_call_fn_2341547�
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
E__inference_dense_24_layer_call_and_return_conditional_losses_2341558�
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
+__inference_output_NN_layer_call_fn_2341567�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2341577�
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
*__inference_Group_NN_layer_call_fn_2339688dense_20_input"�
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
*__inference_Group_NN_layer_call_fn_2339751dense_20_input"�
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
*__inference_Group_NN_layer_call_fn_2341178inputs"�
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
*__inference_Group_NN_layer_call_fn_2341207inputs"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339590dense_20_input"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339624dense_20_input"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341251inputs"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341295inputs"�
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
*__inference_dense_25_layer_call_fn_2341586�
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
E__inference_dense_25_layer_call_and_return_conditional_losses_2341596�
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
*__inference_dense_26_layer_call_fn_2341605�
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2341616�
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
*__inference_dense_27_layer_call_fn_2341625�
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2341636�
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
*__inference_dense_28_layer_call_fn_2341645�
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
E__inference_dense_28_layer_call_and_return_conditional_losses_2341656�
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
*__inference_dense_29_layer_call_fn_2341665�
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2341676�
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
+__inference_output_NN_layer_call_fn_2341685�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2341695�
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
.__inference_Technique_NN_layer_call_fn_2340072dense_25_input"�
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
.__inference_Technique_NN_layer_call_fn_2340135dense_25_input"�
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
.__inference_Technique_NN_layer_call_fn_2341324inputs"�
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
.__inference_Technique_NN_layer_call_fn_2341353inputs"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2339974dense_25_input"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340008dense_25_input"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341397inputs"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341441inputs"�
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
'__inference_dot_2_layer_call_fn_2341447inputs_0inputs_1"�
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
B__inference_dot_2_layer_call_and_return_conditional_losses_2341459inputs_0inputs_1"�
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
':%	� 2Adam/m/dense_20/kernel
':%	� 2Adam/v/dense_20/kernel
 : 2Adam/m/dense_20/bias
 : 2Adam/v/dense_20/bias
&:$  2Adam/m/dense_21/kernel
&:$  2Adam/v/dense_21/kernel
 : 2Adam/m/dense_21/bias
 : 2Adam/v/dense_21/bias
&:$  2Adam/m/dense_22/kernel
&:$  2Adam/v/dense_22/kernel
 : 2Adam/m/dense_22/bias
 : 2Adam/v/dense_22/bias
&:$  2Adam/m/dense_23/kernel
&:$  2Adam/v/dense_23/kernel
 : 2Adam/m/dense_23/bias
 : 2Adam/v/dense_23/bias
&:$  2Adam/m/dense_24/kernel
&:$  2Adam/v/dense_24/kernel
 : 2Adam/m/dense_24/bias
 : 2Adam/v/dense_24/bias
':% 2Adam/m/output_NN/kernel
':% 2Adam/v/output_NN/kernel
!:2Adam/m/output_NN/bias
!:2Adam/v/output_NN/bias
':%	�@2Adam/m/dense_25/kernel
':%	�@2Adam/v/dense_25/kernel
 :@2Adam/m/dense_25/bias
 :@2Adam/v/dense_25/bias
&:$@@2Adam/m/dense_26/kernel
&:$@@2Adam/v/dense_26/kernel
 :@2Adam/m/dense_26/bias
 :@2Adam/v/dense_26/bias
&:$@@2Adam/m/dense_27/kernel
&:$@@2Adam/v/dense_27/kernel
 :@2Adam/m/dense_27/bias
 :@2Adam/v/dense_27/bias
&:$@@2Adam/m/dense_28/kernel
&:$@@2Adam/v/dense_28/kernel
 :@2Adam/m/dense_28/bias
 :@2Adam/v/dense_28/bias
&:$@@2Adam/m/dense_29/kernel
&:$@@2Adam/v/dense_29/kernel
 :@2Adam/m/dense_29/bias
 :@2Adam/v/dense_29/bias
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
*__inference_dense_20_layer_call_fn_2341468inputs"�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_2341478inputs"�
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
*__inference_dense_21_layer_call_fn_2341487inputs"�
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
E__inference_dense_21_layer_call_and_return_conditional_losses_2341498inputs"�
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
*__inference_dense_22_layer_call_fn_2341507inputs"�
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
E__inference_dense_22_layer_call_and_return_conditional_losses_2341518inputs"�
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
*__inference_dense_23_layer_call_fn_2341527inputs"�
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
E__inference_dense_23_layer_call_and_return_conditional_losses_2341538inputs"�
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
*__inference_dense_24_layer_call_fn_2341547inputs"�
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
E__inference_dense_24_layer_call_and_return_conditional_losses_2341558inputs"�
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
+__inference_output_NN_layer_call_fn_2341567inputs"�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2341577inputs"�
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
*__inference_dense_25_layer_call_fn_2341586inputs"�
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
E__inference_dense_25_layer_call_and_return_conditional_losses_2341596inputs"�
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
*__inference_dense_26_layer_call_fn_2341605inputs"�
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2341616inputs"�
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
*__inference_dense_27_layer_call_fn_2341625inputs"�
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2341636inputs"�
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
*__inference_dense_28_layer_call_fn_2341645inputs"�
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
E__inference_dense_28_layer_call_and_return_conditional_losses_2341656inputs"�
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
*__inference_dense_29_layer_call_fn_2341665inputs"�
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
E__inference_dense_29_layer_call_and_return_conditional_losses_2341676inputs"�
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
+__inference_output_NN_layer_call_fn_2341685inputs"�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2341695inputs"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339590~@�=
6�3
)�&
dense_20_input����������
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_Group_NN_layer_call_and_return_conditional_losses_2339624~@�=
6�3
)�&
dense_20_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341251v8�5
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2341295v8�5
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
*__inference_Group_NN_layer_call_fn_2339688s@�=
6�3
)�&
dense_20_input����������
p

 
� "!�
unknown����������
*__inference_Group_NN_layer_call_fn_2339751s@�=
6�3
)�&
dense_20_input����������
p 

 
� "!�
unknown����������
*__inference_Group_NN_layer_call_fn_2341178k8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
*__inference_Group_NN_layer_call_fn_2341207k8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2339974~ !"#$@�=
6�3
)�&
dense_25_input����������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2340008~ !"#$@�=
6�3
)�&
dense_25_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341397v !"#$8�5
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2341441v !"#$8�5
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
.__inference_Technique_NN_layer_call_fn_2340072s !"#$@�=
6�3
)�&
dense_25_input����������
p

 
� "!�
unknown����������
.__inference_Technique_NN_layer_call_fn_2340135s !"#$@�=
6�3
)�&
dense_25_input����������
p 

 
� "!�
unknown����������
.__inference_Technique_NN_layer_call_fn_2341324k !"#$8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
.__inference_Technique_NN_layer_call_fn_2341353k !"#$8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
"__inference__wrapped_model_2339485� !"#$���
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
E__inference_dense_20_layer_call_and_return_conditional_losses_2341478d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_20_layer_call_fn_2341468Y0�-
&�#
!�
inputs����������
� "!�
unknown��������� �
E__inference_dense_21_layer_call_and_return_conditional_losses_2341498c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_21_layer_call_fn_2341487X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_dense_22_layer_call_and_return_conditional_losses_2341518c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_22_layer_call_fn_2341507X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_dense_23_layer_call_and_return_conditional_losses_2341538c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_23_layer_call_fn_2341527X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_dense_24_layer_call_and_return_conditional_losses_2341558c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_24_layer_call_fn_2341547X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_dense_25_layer_call_and_return_conditional_losses_2341596d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_25_layer_call_fn_2341586Y0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
E__inference_dense_26_layer_call_and_return_conditional_losses_2341616c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_26_layer_call_fn_2341605X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
E__inference_dense_27_layer_call_and_return_conditional_losses_2341636c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_27_layer_call_fn_2341625X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
E__inference_dense_28_layer_call_and_return_conditional_losses_2341656c /�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_28_layer_call_fn_2341645X /�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
E__inference_dense_29_layer_call_and_return_conditional_losses_2341676c!"/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_29_layer_call_fn_2341665X!"/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
B__inference_dot_2_layer_call_and_return_conditional_losses_2341459�Z�W
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
'__inference_dot_2_layer_call_fn_2341447Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� "!�
unknown����������
E__inference_model1_2_layer_call_and_return_conditional_losses_2340337� !"#$���
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2340407� !"#$���
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2341043� !"#$���
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
E__inference_model1_2_layer_call_and_return_conditional_losses_2341149� !"#$���
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
*__inference_model1_2_layer_call_fn_2340532� !"#$���
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
*__inference_model1_2_layer_call_fn_2340656� !"#$���
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
*__inference_model1_2_layer_call_fn_2340883� !"#$���
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
*__inference_model1_2_layer_call_fn_2340937� !"#$���
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2341577c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
F__inference_output_NN_layer_call_and_return_conditional_losses_2341695c#$/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
+__inference_output_NN_layer_call_fn_2341567X/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
+__inference_output_NN_layer_call_fn_2341685X#$/�,
%�"
 �
inputs���������@
� "!�
unknown����������
%__inference_signature_wrapper_2340829� !"#$���
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