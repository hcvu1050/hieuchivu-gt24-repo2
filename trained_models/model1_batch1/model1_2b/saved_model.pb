Ϋ
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
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
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
Adam/v/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_17/bias
y
(Adam/v/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_17/bias
y
(Adam/m/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_17/kernel
�
*Adam/v/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_17/kernel
�
*Adam/m/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_16/bias
y
(Adam/v/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_16/bias
y
(Adam/m/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/v/dense_16/kernel
�
*Adam/v/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/m/dense_16/kernel
�
*Adam/m/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/kernel*
_output_shapes

:@@*
dtype0
�
Adam/v/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/v/dense_15/bias
y
(Adam/v/dense_15/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_15/bias*
_output_shapes
:@*
dtype0
�
Adam/m/dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/m/dense_15/bias
y
(Adam/m/dense_15/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_15/bias*
_output_shapes
:@*
dtype0
�
Adam/v/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/v/dense_15/kernel
�
*Adam/v/dense_15/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_15/kernel*
_output_shapes
:	�@*
dtype0
�
Adam/m/dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdam/m/dense_15/kernel
�
*Adam/m/dense_15/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_15/kernel*
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
Adam/v/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_14/bias
y
(Adam/v/dense_14/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_14/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_14/bias
y
(Adam/m/dense_14/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_14/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_14/kernel
�
*Adam/v/dense_14/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_14/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_14/kernel
�
*Adam/m/dense_14/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_14/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_13/bias
y
(Adam/v/dense_13/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_13/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_13/bias
y
(Adam/m/dense_13/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_13/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/v/dense_13/kernel
�
*Adam/v/dense_13/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_13/kernel*
_output_shapes

:  *
dtype0
�
Adam/m/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/m/dense_13/kernel
�
*Adam/m/dense_13/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_13/kernel*
_output_shapes

:  *
dtype0
�
Adam/v/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/v/dense_12/bias
y
(Adam/v/dense_12/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_12/bias*
_output_shapes
: *
dtype0
�
Adam/m/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/m/dense_12/bias
y
(Adam/m/dense_12/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_12/bias*
_output_shapes
: *
dtype0
�
Adam/v/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/v/dense_12/kernel
�
*Adam/v/dense_12/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_12/kernel*
_output_shapes
:	� *
dtype0
�
Adam/m/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *'
shared_nameAdam/m/dense_12/kernel
�
*Adam/m/dense_12/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_12/kernel*
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
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:@*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@@*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:@@*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:@*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
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
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
: *
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:  *
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
: *
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:  *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
: *
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� * 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_Groupserving_default_input_Techniquedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasoutput_NN/kernel_1output_NN/bias_1dense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasoutput_NN/kerneloutput_NN/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2597900

NoOpNoOp
�j
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�i
value�iB�i B�i
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
z
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
15*
z
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
15*
* 
�
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
"trace_0
#trace_1
$trace_2
%trace_3* 
6
&trace_0
'trace_1
(trace_2
)trace_3* 
* 
�
*layer_with_weights-0
*layer-0
+layer_with_weights-1
+layer-1
,layer_with_weights-2
,layer-2
-layer_with_weights-3
-layer-3
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
�
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6layer_with_weights-2
6layer-2
7layer_with_weights-3
7layer-3
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses*
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
�
D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla*

Kserving_default* 
OI
VARIABLE_VALUEdense_12/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_12/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_13/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_13/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_14/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_14/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEoutput_NN/kernel_1&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEoutput_NN/bias_1&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_15/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_15/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_16/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_16/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_17/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_17/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEoutput_NN/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEoutput_NN/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
	1

2*

L0*
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
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

kernel
bias*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

kernel
bias*
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

kernel
bias*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
6
ntrace_0
otrace_1
ptrace_2
qtrace_3* 
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

kernel
bias*
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

kernel
bias*
�
~	variables
trainable_variables
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
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
�non_trainable_variables
�layers
�metrics
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
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
E0
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
�32*
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
�15*
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
�15*
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
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
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
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
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
0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
 
*0
+1
,2
-3*
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
0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
 
40
51
62
73*
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
VARIABLE_VALUEAdam/m/dense_12/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_12/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_12/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_12/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_13/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_13/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_13/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_13/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_14/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_14/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_14/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_14/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/output_NN/kernel_12optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/output_NN/kernel_12optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/output_NN/bias_12optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/output_NN/bias_12optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_15/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_15/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_15/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_15/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_16/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_16/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_16/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_16/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_17/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_17/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_17/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_17/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/output_NN/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/output_NN/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/output_NN/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/output_NN/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasoutput_NN/kernel_1output_NN/bias_1dense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasoutput_NN/kerneloutput_NN/bias	iterationlearning_rateAdam/m/dense_12/kernelAdam/v/dense_12/kernelAdam/m/dense_12/biasAdam/v/dense_12/biasAdam/m/dense_13/kernelAdam/v/dense_13/kernelAdam/m/dense_13/biasAdam/v/dense_13/biasAdam/m/dense_14/kernelAdam/v/dense_14/kernelAdam/m/dense_14/biasAdam/v/dense_14/biasAdam/m/output_NN/kernel_1Adam/v/output_NN/kernel_1Adam/m/output_NN/bias_1Adam/v/output_NN/bias_1Adam/m/dense_15/kernelAdam/v/dense_15/kernelAdam/m/dense_15/biasAdam/v/dense_15/biasAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biasAdam/m/output_NN/kernelAdam/v/output_NN/kernelAdam/m/output_NN/biasAdam/v/output_NN/biastotalcountConst*A
Tin:
826*
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
 __inference__traced_save_2598846
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/biasdense_13/kerneldense_13/biasdense_14/kerneldense_14/biasoutput_NN/kernel_1output_NN/bias_1dense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasoutput_NN/kerneloutput_NN/bias	iterationlearning_rateAdam/m/dense_12/kernelAdam/v/dense_12/kernelAdam/m/dense_12/biasAdam/v/dense_12/biasAdam/m/dense_13/kernelAdam/v/dense_13/kernelAdam/m/dense_13/biasAdam/v/dense_13/biasAdam/m/dense_14/kernelAdam/v/dense_14/kernelAdam/m/dense_14/biasAdam/v/dense_14/biasAdam/m/output_NN/kernel_1Adam/v/output_NN/kernel_1Adam/m/output_NN/bias_1Adam/v/output_NN/bias_1Adam/m/dense_15/kernelAdam/v/dense_15/kernelAdam/m/dense_15/biasAdam/v/dense_15/biasAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biasAdam/m/output_NN/kernelAdam/v/output_NN/kernelAdam/m/output_NN/biasAdam/v/output_NN/biastotalcount*@
Tin9
725*
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
#__inference__traced_restore_2599012��
�

�
E__inference_dense_16_layer_call_and_return_conditional_losses_2598471

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
�
*__inference_model1_4_layer_call_fn_2597775
input_group
input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_groupinput_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model1_4_layer_call_and_return_conditional_losses_2597740o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 22
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
*__inference_dense_12_layer_call_fn_2598363

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
E__inference_dense_12_layer_call_and_return_conditional_losses_2596946o
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
�
.__inference_Technique_NN_layer_call_fn_2597341
dense_15_input
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_15_input
�
�
*__inference_model1_4_layer_call_fn_2597938
inputs_input_group
inputs_input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_input_groupinputs_input_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model1_4_layer_call_and_return_conditional_losses_2597648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 22
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2596946

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
F__inference_output_NN_layer_call_and_return_conditional_losses_2596996

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
�
.__inference_Technique_NN_layer_call_fn_2598255

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597322o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
E__inference_dense_14_layer_call_and_return_conditional_losses_2598413

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
E__inference_dense_15_layer_call_and_return_conditional_losses_2597214

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
*__inference_dense_15_layer_call_fn_2598441

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
E__inference_dense_15_layer_call_and_return_conditional_losses_2597214o
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
��
� 
#__inference__traced_restore_2599012
file_prefix3
 assignvariableop_dense_12_kernel:	� .
 assignvariableop_1_dense_12_bias: 4
"assignvariableop_2_dense_13_kernel:  .
 assignvariableop_3_dense_13_bias: 4
"assignvariableop_4_dense_14_kernel:  .
 assignvariableop_5_dense_14_bias: 7
%assignvariableop_6_output_nn_kernel_1: 1
#assignvariableop_7_output_nn_bias_1:5
"assignvariableop_8_dense_15_kernel:	�@.
 assignvariableop_9_dense_15_bias:@5
#assignvariableop_10_dense_16_kernel:@@/
!assignvariableop_11_dense_16_bias:@5
#assignvariableop_12_dense_17_kernel:@@/
!assignvariableop_13_dense_17_bias:@6
$assignvariableop_14_output_nn_kernel:@0
"assignvariableop_15_output_nn_bias:'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: =
*assignvariableop_18_adam_m_dense_12_kernel:	� =
*assignvariableop_19_adam_v_dense_12_kernel:	� 6
(assignvariableop_20_adam_m_dense_12_bias: 6
(assignvariableop_21_adam_v_dense_12_bias: <
*assignvariableop_22_adam_m_dense_13_kernel:  <
*assignvariableop_23_adam_v_dense_13_kernel:  6
(assignvariableop_24_adam_m_dense_13_bias: 6
(assignvariableop_25_adam_v_dense_13_bias: <
*assignvariableop_26_adam_m_dense_14_kernel:  <
*assignvariableop_27_adam_v_dense_14_kernel:  6
(assignvariableop_28_adam_m_dense_14_bias: 6
(assignvariableop_29_adam_v_dense_14_bias: ?
-assignvariableop_30_adam_m_output_nn_kernel_1: ?
-assignvariableop_31_adam_v_output_nn_kernel_1: 9
+assignvariableop_32_adam_m_output_nn_bias_1:9
+assignvariableop_33_adam_v_output_nn_bias_1:=
*assignvariableop_34_adam_m_dense_15_kernel:	�@=
*assignvariableop_35_adam_v_dense_15_kernel:	�@6
(assignvariableop_36_adam_m_dense_15_bias:@6
(assignvariableop_37_adam_v_dense_15_bias:@<
*assignvariableop_38_adam_m_dense_16_kernel:@@<
*assignvariableop_39_adam_v_dense_16_kernel:@@6
(assignvariableop_40_adam_m_dense_16_bias:@6
(assignvariableop_41_adam_v_dense_16_bias:@<
*assignvariableop_42_adam_m_dense_17_kernel:@@<
*assignvariableop_43_adam_v_dense_17_kernel:@@6
(assignvariableop_44_adam_m_dense_17_bias:@6
(assignvariableop_45_adam_v_dense_17_bias:@=
+assignvariableop_46_adam_m_output_nn_kernel:@=
+assignvariableop_47_adam_v_output_nn_kernel:@7
)assignvariableop_48_adam_m_output_nn_bias:7
)assignvariableop_49_adam_v_output_nn_bias:#
assignvariableop_50_total: #
assignvariableop_51_count: 
identity_53��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_13_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_13_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_14_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_14_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_output_nn_kernel_1Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_output_nn_bias_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_15_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_15_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_16_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_16_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_17_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_17_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_output_nn_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp"assignvariableop_15_output_nn_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_m_dense_12_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_v_dense_12_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_m_dense_12_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_v_dense_12_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_dense_13_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_dense_13_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_m_dense_13_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_v_dense_13_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_14_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_14_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_m_dense_14_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_v_dense_14_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_m_output_nn_kernel_1Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_v_output_nn_kernel_1Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_m_output_nn_bias_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_v_output_nn_bias_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_15_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_15_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_m_dense_15_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp(assignvariableop_37_adam_v_dense_15_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_16_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_16_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_m_dense_16_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_v_dense_16_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_17_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_17_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_m_dense_17_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_v_dense_17_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_m_output_nn_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_v_output_nn_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_m_output_nn_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_v_output_nn_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�#
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598234

inputs:
'dense_12_matmul_readvariableop_resource:	� 6
(dense_12_biasadd_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource:  6
(dense_13_biasadd_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource:  6
(dense_14_biasadd_readvariableop_resource: :
(output_nn_matmul_readvariableop_resource: 7
)output_nn_biasadd_readvariableop_resource:
identity��dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0{
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_13/MatMulMatMuldense_12/BiasAdd:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output_NN/MatMulMatMuldense_14/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597003
dense_12_input#
dense_12_2596947:	� 
dense_12_2596949: "
dense_13_2596964:  
dense_13_2596966: "
dense_14_2596981:  
dense_14_2596983: #
output_nn_2596997: 
output_nn_2596999:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_2596947dense_12_2596949*
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2596946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2596964dense_13_2596966*
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
E__inference_dense_13_layer_call_and_return_conditional_losses_2596963�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_2596981dense_14_2596983*
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2596980�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0output_nn_2596997output_nn_2596999*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2596996y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_12_input
�	
�
*__inference_Group_NN_layer_call_fn_2598153

inputs
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597054o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
F__inference_output_NN_layer_call_and_return_conditional_losses_2598510

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
�
�
*__inference_dense_14_layer_call_fn_2598402

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
E__inference_dense_14_layer_call_and_return_conditional_losses_2596980o
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
�$
�
E__inference_model1_4_layer_call_and_return_conditional_losses_2597590
input_group
input_technique#
group_nn_2597540:	� 
group_nn_2597542: "
group_nn_2597544:  
group_nn_2597546: "
group_nn_2597548:  
group_nn_2597550: "
group_nn_2597552: 
group_nn_2597554:'
technique_nn_2597557:	�@"
technique_nn_2597559:@&
technique_nn_2597561:@@"
technique_nn_2597563:@&
technique_nn_2597565:@@"
technique_nn_2597567:@&
technique_nn_2597569:@"
technique_nn_2597571:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinput_groupgroup_nn_2597540group_nn_2597542group_nn_2597544group_nn_2597546group_nn_2597548group_nn_2597550group_nn_2597552group_nn_2597554*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597099�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinput_techniquetechnique_nn_2597557technique_nn_2597559technique_nn_2597561technique_nn_2597563technique_nn_2597565technique_nn_2597567technique_nn_2597569technique_nn_2597571*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597367z
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
dot_4/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2597533m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2D
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
*__inference_dense_17_layer_call_fn_2598480

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
E__inference_dense_17_layer_call_and_return_conditional_losses_2597248o
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
�
*__inference_Group_NN_layer_call_fn_2598174

inputs
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
E__inference_model1_4_layer_call_and_return_conditional_losses_2597536
input_group
input_technique#
group_nn_2597473:	� 
group_nn_2597475: "
group_nn_2597477:  
group_nn_2597479: "
group_nn_2597481:  
group_nn_2597483: "
group_nn_2597485: 
group_nn_2597487:'
technique_nn_2597490:	�@"
technique_nn_2597492:@&
technique_nn_2597494:@@"
technique_nn_2597496:@&
technique_nn_2597498:@@"
technique_nn_2597500:@&
technique_nn_2597502:@"
technique_nn_2597504:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinput_groupgroup_nn_2597473group_nn_2597475group_nn_2597477group_nn_2597479group_nn_2597481group_nn_2597483group_nn_2597485group_nn_2597487*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597054�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinput_techniquetechnique_nn_2597490technique_nn_2597492technique_nn_2597494technique_nn_2597496technique_nn_2597498technique_nn_2597500technique_nn_2597502technique_nn_2597504*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597322z
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
dot_4/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2597533m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2D
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
*__inference_dense_16_layer_call_fn_2598460

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
E__inference_dense_16_layer_call_and_return_conditional_losses_2597231o
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2596980

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
E__inference_dense_13_layer_call_and_return_conditional_losses_2598393

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
�#
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598336

inputs:
'dense_15_matmul_readvariableop_resource:	�@6
(dense_15_biasadd_readvariableop_resource:@9
'dense_16_matmul_readvariableop_resource:@@6
(dense_16_biasadd_readvariableop_resource:@9
'dense_17_matmul_readvariableop_resource:@@6
(dense_17_biasadd_readvariableop_resource:@:
(output_nn_matmul_readvariableop_resource:@7
)output_nn_biasadd_readvariableop_resource:
identity��dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0{
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_16/MatMulMatMuldense_15/BiasAdd:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
output_NN/MatMulMatMuldense_17/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model1_4_layer_call_fn_2597683
input_group
input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_groupinput_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model1_4_layer_call_and_return_conditional_losses_2597648o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 22
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
�$
�
E__inference_model1_4_layer_call_and_return_conditional_losses_2597740

inputs
inputs_1#
group_nn_2597690:	� 
group_nn_2597692: "
group_nn_2597694:  
group_nn_2597696: "
group_nn_2597698:  
group_nn_2597700: "
group_nn_2597702: 
group_nn_2597704:'
technique_nn_2597707:	�@"
technique_nn_2597709:@&
technique_nn_2597711:@@"
technique_nn_2597713:@&
technique_nn_2597715:@@"
technique_nn_2597717:@&
technique_nn_2597719:@"
technique_nn_2597721:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinputsgroup_nn_2597690group_nn_2597692group_nn_2597694group_nn_2597696group_nn_2597698group_nn_2597700group_nn_2597702group_nn_2597704*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597099�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinputs_1technique_nn_2597707technique_nn_2597709technique_nn_2597711technique_nn_2597713technique_nn_2597715technique_nn_2597717technique_nn_2597719technique_nn_2597721*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597367z
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
dot_4/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2597533m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2D
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
l
B__inference_dot_4_layer_call_and_return_conditional_losses_2597533

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
�
F__inference_output_NN_layer_call_and_return_conditional_losses_2597264

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
�
.__inference_Technique_NN_layer_call_fn_2598276

inputs
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597367o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�f
�
E__inference_model1_4_layer_call_and_return_conditional_losses_2598054
inputs_input_group
inputs_input_techniqueC
0group_nn_dense_12_matmul_readvariableop_resource:	� ?
1group_nn_dense_12_biasadd_readvariableop_resource: B
0group_nn_dense_13_matmul_readvariableop_resource:  ?
1group_nn_dense_13_biasadd_readvariableop_resource: B
0group_nn_dense_14_matmul_readvariableop_resource:  ?
1group_nn_dense_14_biasadd_readvariableop_resource: C
1group_nn_output_nn_matmul_readvariableop_resource: @
2group_nn_output_nn_biasadd_readvariableop_resource:G
4technique_nn_dense_15_matmul_readvariableop_resource:	�@C
5technique_nn_dense_15_biasadd_readvariableop_resource:@F
4technique_nn_dense_16_matmul_readvariableop_resource:@@C
5technique_nn_dense_16_biasadd_readvariableop_resource:@F
4technique_nn_dense_17_matmul_readvariableop_resource:@@C
5technique_nn_dense_17_biasadd_readvariableop_resource:@G
5technique_nn_output_nn_matmul_readvariableop_resource:@D
6technique_nn_output_nn_biasadd_readvariableop_resource:
identity��(Group_NN/dense_12/BiasAdd/ReadVariableOp�'Group_NN/dense_12/MatMul/ReadVariableOp�(Group_NN/dense_13/BiasAdd/ReadVariableOp�'Group_NN/dense_13/MatMul/ReadVariableOp�(Group_NN/dense_14/BiasAdd/ReadVariableOp�'Group_NN/dense_14/MatMul/ReadVariableOp�)Group_NN/output_NN/BiasAdd/ReadVariableOp�(Group_NN/output_NN/MatMul/ReadVariableOp�,Technique_NN/dense_15/BiasAdd/ReadVariableOp�+Technique_NN/dense_15/MatMul/ReadVariableOp�,Technique_NN/dense_16/BiasAdd/ReadVariableOp�+Technique_NN/dense_16/MatMul/ReadVariableOp�,Technique_NN/dense_17/BiasAdd/ReadVariableOp�+Technique_NN/dense_17/MatMul/ReadVariableOp�-Technique_NN/output_NN/BiasAdd/ReadVariableOp�,Technique_NN/output_NN/MatMul/ReadVariableOp�
'Group_NN/dense_12/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_12_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
Group_NN/dense_12/MatMulMatMulinputs_input_group/Group_NN/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_12/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_12/BiasAddBiasAdd"Group_NN/dense_12/MatMul:product:00Group_NN/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_13/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_13_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_13/MatMulMatMul"Group_NN/dense_12/BiasAdd:output:0/Group_NN/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_13/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_13/BiasAddBiasAdd"Group_NN/dense_13/MatMul:product:00Group_NN/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_13/ReluRelu"Group_NN/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_14/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_14/MatMulMatMul$Group_NN/dense_13/Relu:activations:0/Group_NN/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_14/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_14/BiasAddBiasAdd"Group_NN/dense_14/MatMul:product:00Group_NN/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_14/ReluRelu"Group_NN/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp1group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Group_NN/output_NN/MatMulMatMul$Group_NN/dense_14/Relu:activations:00Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
+Technique_NN/dense_15/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Technique_NN/dense_15/MatMulMatMulinputs_input_technique3Technique_NN/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_15/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_15/BiasAddBiasAdd&Technique_NN/dense_15/MatMul:product:04Technique_NN/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_16/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_16/MatMulMatMul&Technique_NN/dense_15/BiasAdd:output:03Technique_NN/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_16/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_16/BiasAddBiasAdd&Technique_NN/dense_16/MatMul:product:04Technique_NN/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_16/ReluRelu&Technique_NN/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_17/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_17/MatMulMatMul(Technique_NN/dense_16/Relu:activations:03Technique_NN/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_17/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_17/BiasAddBiasAdd&Technique_NN/dense_17/MatMul:product:04Technique_NN/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_17/ReluRelu&Technique_NN/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp5technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
Technique_NN/output_NN/MatMulMatMul(Technique_NN/dense_17/Relu:activations:04Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
dot_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_4/ExpandDims
ExpandDimsl2_normalize:z:0dot_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������X
dot_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_4/ExpandDims_1
ExpandDimsl2_normalize_1:z:0dot_4/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
dot_4/MatMulBatchMatMulV2dot_4/ExpandDims:output:0dot_4/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������^
dot_4/ShapeShapedot_4/MatMul:output:0*
T0*
_output_shapes
::��x
dot_4/SqueezeSqueezedot_4/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
e
IdentityIdentitydot_4/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^Group_NN/dense_12/BiasAdd/ReadVariableOp(^Group_NN/dense_12/MatMul/ReadVariableOp)^Group_NN/dense_13/BiasAdd/ReadVariableOp(^Group_NN/dense_13/MatMul/ReadVariableOp)^Group_NN/dense_14/BiasAdd/ReadVariableOp(^Group_NN/dense_14/MatMul/ReadVariableOp*^Group_NN/output_NN/BiasAdd/ReadVariableOp)^Group_NN/output_NN/MatMul/ReadVariableOp-^Technique_NN/dense_15/BiasAdd/ReadVariableOp,^Technique_NN/dense_15/MatMul/ReadVariableOp-^Technique_NN/dense_16/BiasAdd/ReadVariableOp,^Technique_NN/dense_16/MatMul/ReadVariableOp-^Technique_NN/dense_17/BiasAdd/ReadVariableOp,^Technique_NN/dense_17/MatMul/ReadVariableOp.^Technique_NN/output_NN/BiasAdd/ReadVariableOp-^Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2T
(Group_NN/dense_12/BiasAdd/ReadVariableOp(Group_NN/dense_12/BiasAdd/ReadVariableOp2R
'Group_NN/dense_12/MatMul/ReadVariableOp'Group_NN/dense_12/MatMul/ReadVariableOp2T
(Group_NN/dense_13/BiasAdd/ReadVariableOp(Group_NN/dense_13/BiasAdd/ReadVariableOp2R
'Group_NN/dense_13/MatMul/ReadVariableOp'Group_NN/dense_13/MatMul/ReadVariableOp2T
(Group_NN/dense_14/BiasAdd/ReadVariableOp(Group_NN/dense_14/BiasAdd/ReadVariableOp2R
'Group_NN/dense_14/MatMul/ReadVariableOp'Group_NN/dense_14/MatMul/ReadVariableOp2V
)Group_NN/output_NN/BiasAdd/ReadVariableOp)Group_NN/output_NN/BiasAdd/ReadVariableOp2T
(Group_NN/output_NN/MatMul/ReadVariableOp(Group_NN/output_NN/MatMul/ReadVariableOp2\
,Technique_NN/dense_15/BiasAdd/ReadVariableOp,Technique_NN/dense_15/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_15/MatMul/ReadVariableOp+Technique_NN/dense_15/MatMul/ReadVariableOp2\
,Technique_NN/dense_16/BiasAdd/ReadVariableOp,Technique_NN/dense_16/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_16/MatMul/ReadVariableOp+Technique_NN/dense_16/MatMul/ReadVariableOp2\
,Technique_NN/dense_17/BiasAdd/ReadVariableOp,Technique_NN/dense_17/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_17/MatMul/ReadVariableOp+Technique_NN/dense_17/MatMul/ReadVariableOp2^
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
�
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597099

inputs#
dense_12_2597078:	� 
dense_12_2597080: "
dense_13_2597083:  
dense_13_2597085: "
dense_14_2597088:  
dense_14_2597090: #
output_nn_2597093: 
output_nn_2597095:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_2597078dense_12_2597080*
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2596946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2597083dense_13_2597085*
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
E__inference_dense_13_layer_call_and_return_conditional_losses_2596963�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_2597088dense_14_2597090*
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2596980�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0output_nn_2597093output_nn_2597095*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2596996y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597322

inputs#
dense_15_2597301:	�@
dense_15_2597303:@"
dense_16_2597306:@@
dense_16_2597308:@"
dense_17_2597311:@@
dense_17_2597313:@#
output_nn_2597316:@
output_nn_2597318:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_2597301dense_15_2597303*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2597214�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_2597306dense_16_2597308*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2597231�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_2597311dense_17_2597313*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_2597248�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0output_nn_2597316output_nn_2597318*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2597264y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�t
�
"__inference__wrapped_model_2596932
input_group
input_techniqueL
9model1_4_group_nn_dense_12_matmul_readvariableop_resource:	� H
:model1_4_group_nn_dense_12_biasadd_readvariableop_resource: K
9model1_4_group_nn_dense_13_matmul_readvariableop_resource:  H
:model1_4_group_nn_dense_13_biasadd_readvariableop_resource: K
9model1_4_group_nn_dense_14_matmul_readvariableop_resource:  H
:model1_4_group_nn_dense_14_biasadd_readvariableop_resource: L
:model1_4_group_nn_output_nn_matmul_readvariableop_resource: I
;model1_4_group_nn_output_nn_biasadd_readvariableop_resource:P
=model1_4_technique_nn_dense_15_matmul_readvariableop_resource:	�@L
>model1_4_technique_nn_dense_15_biasadd_readvariableop_resource:@O
=model1_4_technique_nn_dense_16_matmul_readvariableop_resource:@@L
>model1_4_technique_nn_dense_16_biasadd_readvariableop_resource:@O
=model1_4_technique_nn_dense_17_matmul_readvariableop_resource:@@L
>model1_4_technique_nn_dense_17_biasadd_readvariableop_resource:@P
>model1_4_technique_nn_output_nn_matmul_readvariableop_resource:@M
?model1_4_technique_nn_output_nn_biasadd_readvariableop_resource:
identity��1model1_4/Group_NN/dense_12/BiasAdd/ReadVariableOp�0model1_4/Group_NN/dense_12/MatMul/ReadVariableOp�1model1_4/Group_NN/dense_13/BiasAdd/ReadVariableOp�0model1_4/Group_NN/dense_13/MatMul/ReadVariableOp�1model1_4/Group_NN/dense_14/BiasAdd/ReadVariableOp�0model1_4/Group_NN/dense_14/MatMul/ReadVariableOp�2model1_4/Group_NN/output_NN/BiasAdd/ReadVariableOp�1model1_4/Group_NN/output_NN/MatMul/ReadVariableOp�5model1_4/Technique_NN/dense_15/BiasAdd/ReadVariableOp�4model1_4/Technique_NN/dense_15/MatMul/ReadVariableOp�5model1_4/Technique_NN/dense_16/BiasAdd/ReadVariableOp�4model1_4/Technique_NN/dense_16/MatMul/ReadVariableOp�5model1_4/Technique_NN/dense_17/BiasAdd/ReadVariableOp�4model1_4/Technique_NN/dense_17/MatMul/ReadVariableOp�6model1_4/Technique_NN/output_NN/BiasAdd/ReadVariableOp�5model1_4/Technique_NN/output_NN/MatMul/ReadVariableOp�
0model1_4/Group_NN/dense_12/MatMul/ReadVariableOpReadVariableOp9model1_4_group_nn_dense_12_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
!model1_4/Group_NN/dense_12/MatMulMatMulinput_group8model1_4/Group_NN/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_4/Group_NN/dense_12/BiasAdd/ReadVariableOpReadVariableOp:model1_4_group_nn_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_4/Group_NN/dense_12/BiasAddBiasAdd+model1_4/Group_NN/dense_12/MatMul:product:09model1_4/Group_NN/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
0model1_4/Group_NN/dense_13/MatMul/ReadVariableOpReadVariableOp9model1_4_group_nn_dense_13_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
!model1_4/Group_NN/dense_13/MatMulMatMul+model1_4/Group_NN/dense_12/BiasAdd:output:08model1_4/Group_NN/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_4/Group_NN/dense_13/BiasAdd/ReadVariableOpReadVariableOp:model1_4_group_nn_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_4/Group_NN/dense_13/BiasAddBiasAdd+model1_4/Group_NN/dense_13/MatMul:product:09model1_4/Group_NN/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1_4/Group_NN/dense_13/ReluRelu+model1_4/Group_NN/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
0model1_4/Group_NN/dense_14/MatMul/ReadVariableOpReadVariableOp9model1_4_group_nn_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
!model1_4/Group_NN/dense_14/MatMulMatMul-model1_4/Group_NN/dense_13/Relu:activations:08model1_4/Group_NN/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1model1_4/Group_NN/dense_14/BiasAdd/ReadVariableOpReadVariableOp:model1_4_group_nn_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"model1_4/Group_NN/dense_14/BiasAddBiasAdd+model1_4/Group_NN/dense_14/MatMul:product:09model1_4/Group_NN/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
model1_4/Group_NN/dense_14/ReluRelu+model1_4/Group_NN/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
1model1_4/Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp:model1_4_group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
"model1_4/Group_NN/output_NN/MatMulMatMul-model1_4/Group_NN/dense_14/Relu:activations:09model1_4/Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2model1_4/Group_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp;model1_4_group_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model1_4/Group_NN/output_NN/BiasAddBiasAdd,model1_4/Group_NN/output_NN/MatMul:product:0:model1_4/Group_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
4model1_4/Technique_NN/dense_15/MatMul/ReadVariableOpReadVariableOp=model1_4_technique_nn_dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
%model1_4/Technique_NN/dense_15/MatMulMatMulinput_technique<model1_4/Technique_NN/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_4/Technique_NN/dense_15/BiasAdd/ReadVariableOpReadVariableOp>model1_4_technique_nn_dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_4/Technique_NN/dense_15/BiasAddBiasAdd/model1_4/Technique_NN/dense_15/MatMul:product:0=model1_4/Technique_NN/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
4model1_4/Technique_NN/dense_16/MatMul/ReadVariableOpReadVariableOp=model1_4_technique_nn_dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
%model1_4/Technique_NN/dense_16/MatMulMatMul/model1_4/Technique_NN/dense_15/BiasAdd:output:0<model1_4/Technique_NN/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_4/Technique_NN/dense_16/BiasAdd/ReadVariableOpReadVariableOp>model1_4_technique_nn_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_4/Technique_NN/dense_16/BiasAddBiasAdd/model1_4/Technique_NN/dense_16/MatMul:product:0=model1_4/Technique_NN/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model1_4/Technique_NN/dense_16/ReluRelu/model1_4/Technique_NN/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
4model1_4/Technique_NN/dense_17/MatMul/ReadVariableOpReadVariableOp=model1_4_technique_nn_dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
%model1_4/Technique_NN/dense_17/MatMulMatMul1model1_4/Technique_NN/dense_16/Relu:activations:0<model1_4/Technique_NN/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
5model1_4/Technique_NN/dense_17/BiasAdd/ReadVariableOpReadVariableOp>model1_4_technique_nn_dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
&model1_4/Technique_NN/dense_17/BiasAddBiasAdd/model1_4/Technique_NN/dense_17/MatMul:product:0=model1_4/Technique_NN/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model1_4/Technique_NN/dense_17/ReluRelu/model1_4/Technique_NN/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
5model1_4/Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp>model1_4_technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model1_4/Technique_NN/output_NN/MatMulMatMul1model1_4/Technique_NN/dense_17/Relu:activations:0=model1_4/Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6model1_4/Technique_NN/output_NN/BiasAdd/ReadVariableOpReadVariableOp?model1_4_technique_nn_output_nn_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
'model1_4/Technique_NN/output_NN/BiasAddBiasAdd0model1_4/Technique_NN/output_NN/MatMul:product:0>model1_4/Technique_NN/output_NN/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model1_4/l2_normalize/SquareSquare,model1_4/Group_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������m
+model1_4/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model1_4/l2_normalize/SumSum model1_4/l2_normalize/Square:y:04model1_4/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(d
model1_4/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
model1_4/l2_normalize/MaximumMaximum"model1_4/l2_normalize/Sum:output:0(model1_4/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:���������y
model1_4/l2_normalize/RsqrtRsqrt!model1_4/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:����������
model1_4/l2_normalizeMul,model1_4/Group_NN/output_NN/BiasAdd:output:0model1_4/l2_normalize/Rsqrt:y:0*
T0*'
_output_shapes
:����������
model1_4/l2_normalize_1/SquareSquare0model1_4/Technique_NN/output_NN/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
-model1_4/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model1_4/l2_normalize_1/SumSum"model1_4/l2_normalize_1/Square:y:06model1_4/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(f
!model1_4/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼�+�
model1_4/l2_normalize_1/MaximumMaximum$model1_4/l2_normalize_1/Sum:output:0*model1_4/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:���������}
model1_4/l2_normalize_1/RsqrtRsqrt#model1_4/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:����������
model1_4/l2_normalize_1Mul0model1_4/Technique_NN/output_NN/BiasAdd:output:0!model1_4/l2_normalize_1/Rsqrt:y:0*
T0*'
_output_shapes
:���������_
model1_4/dot_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model1_4/dot_4/ExpandDims
ExpandDimsmodel1_4/l2_normalize:z:0&model1_4/dot_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������a
model1_4/dot_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model1_4/dot_4/ExpandDims_1
ExpandDimsmodel1_4/l2_normalize_1:z:0(model1_4/dot_4/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
model1_4/dot_4/MatMulBatchMatMulV2"model1_4/dot_4/ExpandDims:output:0$model1_4/dot_4/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������p
model1_4/dot_4/ShapeShapemodel1_4/dot_4/MatMul:output:0*
T0*
_output_shapes
::���
model1_4/dot_4/SqueezeSqueezemodel1_4/dot_4/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
n
IdentityIdentitymodel1_4/dot_4/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp2^model1_4/Group_NN/dense_12/BiasAdd/ReadVariableOp1^model1_4/Group_NN/dense_12/MatMul/ReadVariableOp2^model1_4/Group_NN/dense_13/BiasAdd/ReadVariableOp1^model1_4/Group_NN/dense_13/MatMul/ReadVariableOp2^model1_4/Group_NN/dense_14/BiasAdd/ReadVariableOp1^model1_4/Group_NN/dense_14/MatMul/ReadVariableOp3^model1_4/Group_NN/output_NN/BiasAdd/ReadVariableOp2^model1_4/Group_NN/output_NN/MatMul/ReadVariableOp6^model1_4/Technique_NN/dense_15/BiasAdd/ReadVariableOp5^model1_4/Technique_NN/dense_15/MatMul/ReadVariableOp6^model1_4/Technique_NN/dense_16/BiasAdd/ReadVariableOp5^model1_4/Technique_NN/dense_16/MatMul/ReadVariableOp6^model1_4/Technique_NN/dense_17/BiasAdd/ReadVariableOp5^model1_4/Technique_NN/dense_17/MatMul/ReadVariableOp7^model1_4/Technique_NN/output_NN/BiasAdd/ReadVariableOp6^model1_4/Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2f
1model1_4/Group_NN/dense_12/BiasAdd/ReadVariableOp1model1_4/Group_NN/dense_12/BiasAdd/ReadVariableOp2d
0model1_4/Group_NN/dense_12/MatMul/ReadVariableOp0model1_4/Group_NN/dense_12/MatMul/ReadVariableOp2f
1model1_4/Group_NN/dense_13/BiasAdd/ReadVariableOp1model1_4/Group_NN/dense_13/BiasAdd/ReadVariableOp2d
0model1_4/Group_NN/dense_13/MatMul/ReadVariableOp0model1_4/Group_NN/dense_13/MatMul/ReadVariableOp2f
1model1_4/Group_NN/dense_14/BiasAdd/ReadVariableOp1model1_4/Group_NN/dense_14/BiasAdd/ReadVariableOp2d
0model1_4/Group_NN/dense_14/MatMul/ReadVariableOp0model1_4/Group_NN/dense_14/MatMul/ReadVariableOp2h
2model1_4/Group_NN/output_NN/BiasAdd/ReadVariableOp2model1_4/Group_NN/output_NN/BiasAdd/ReadVariableOp2f
1model1_4/Group_NN/output_NN/MatMul/ReadVariableOp1model1_4/Group_NN/output_NN/MatMul/ReadVariableOp2n
5model1_4/Technique_NN/dense_15/BiasAdd/ReadVariableOp5model1_4/Technique_NN/dense_15/BiasAdd/ReadVariableOp2l
4model1_4/Technique_NN/dense_15/MatMul/ReadVariableOp4model1_4/Technique_NN/dense_15/MatMul/ReadVariableOp2n
5model1_4/Technique_NN/dense_16/BiasAdd/ReadVariableOp5model1_4/Technique_NN/dense_16/BiasAdd/ReadVariableOp2l
4model1_4/Technique_NN/dense_16/MatMul/ReadVariableOp4model1_4/Technique_NN/dense_16/MatMul/ReadVariableOp2n
5model1_4/Technique_NN/dense_17/BiasAdd/ReadVariableOp5model1_4/Technique_NN/dense_17/BiasAdd/ReadVariableOp2l
4model1_4/Technique_NN/dense_17/MatMul/ReadVariableOp4model1_4/Technique_NN/dense_17/MatMul/ReadVariableOp2p
6model1_4/Technique_NN/output_NN/BiasAdd/ReadVariableOp6model1_4/Technique_NN/output_NN/BiasAdd/ReadVariableOp2n
5model1_4/Technique_NN/output_NN/MatMul/ReadVariableOp5model1_4/Technique_NN/output_NN/MatMul/ReadVariableOp:YU
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2597231

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
�
%__inference_signature_wrapper_2597900
input_group
input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_groupinput_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2596932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 22
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
+__inference_output_NN_layer_call_fn_2598500

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
F__inference_output_NN_layer_call_and_return_conditional_losses_2597264o
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2598373

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
�
.__inference_Technique_NN_layer_call_fn_2597386
dense_15_input
unknown:	�@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597367o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_15_input
�
�
*__inference_model1_4_layer_call_fn_2597976
inputs_input_group
inputs_input_technique
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_input_groupinputs_input_techniqueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model1_4_layer_call_and_return_conditional_losses_2597740o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 22
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
��
�/
 __inference__traced_save_2598846
file_prefix9
&read_disablecopyonread_dense_12_kernel:	� 4
&read_1_disablecopyonread_dense_12_bias: :
(read_2_disablecopyonread_dense_13_kernel:  4
&read_3_disablecopyonread_dense_13_bias: :
(read_4_disablecopyonread_dense_14_kernel:  4
&read_5_disablecopyonread_dense_14_bias: =
+read_6_disablecopyonread_output_nn_kernel_1: 7
)read_7_disablecopyonread_output_nn_bias_1:;
(read_8_disablecopyonread_dense_15_kernel:	�@4
&read_9_disablecopyonread_dense_15_bias:@;
)read_10_disablecopyonread_dense_16_kernel:@@5
'read_11_disablecopyonread_dense_16_bias:@;
)read_12_disablecopyonread_dense_17_kernel:@@5
'read_13_disablecopyonread_dense_17_bias:@<
*read_14_disablecopyonread_output_nn_kernel:@6
(read_15_disablecopyonread_output_nn_bias:-
#read_16_disablecopyonread_iteration:	 1
'read_17_disablecopyonread_learning_rate: C
0read_18_disablecopyonread_adam_m_dense_12_kernel:	� C
0read_19_disablecopyonread_adam_v_dense_12_kernel:	� <
.read_20_disablecopyonread_adam_m_dense_12_bias: <
.read_21_disablecopyonread_adam_v_dense_12_bias: B
0read_22_disablecopyonread_adam_m_dense_13_kernel:  B
0read_23_disablecopyonread_adam_v_dense_13_kernel:  <
.read_24_disablecopyonread_adam_m_dense_13_bias: <
.read_25_disablecopyonread_adam_v_dense_13_bias: B
0read_26_disablecopyonread_adam_m_dense_14_kernel:  B
0read_27_disablecopyonread_adam_v_dense_14_kernel:  <
.read_28_disablecopyonread_adam_m_dense_14_bias: <
.read_29_disablecopyonread_adam_v_dense_14_bias: E
3read_30_disablecopyonread_adam_m_output_nn_kernel_1: E
3read_31_disablecopyonread_adam_v_output_nn_kernel_1: ?
1read_32_disablecopyonread_adam_m_output_nn_bias_1:?
1read_33_disablecopyonread_adam_v_output_nn_bias_1:C
0read_34_disablecopyonread_adam_m_dense_15_kernel:	�@C
0read_35_disablecopyonread_adam_v_dense_15_kernel:	�@<
.read_36_disablecopyonread_adam_m_dense_15_bias:@<
.read_37_disablecopyonread_adam_v_dense_15_bias:@B
0read_38_disablecopyonread_adam_m_dense_16_kernel:@@B
0read_39_disablecopyonread_adam_v_dense_16_kernel:@@<
.read_40_disablecopyonread_adam_m_dense_16_bias:@<
.read_41_disablecopyonread_adam_v_dense_16_bias:@B
0read_42_disablecopyonread_adam_m_dense_17_kernel:@@B
0read_43_disablecopyonread_adam_v_dense_17_kernel:@@<
.read_44_disablecopyonread_adam_m_dense_17_bias:@<
.read_45_disablecopyonread_adam_v_dense_17_bias:@C
1read_46_disablecopyonread_adam_m_output_nn_kernel:@C
1read_47_disablecopyonread_adam_v_output_nn_kernel:@=
/read_48_disablecopyonread_adam_m_output_nn_bias:=
/read_49_disablecopyonread_adam_v_output_nn_bias:)
read_50_disablecopyonread_total: )
read_51_disablecopyonread_count: 
savev2_const
identity_105��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_12_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_12_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_13_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_13_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_14_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_14_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
: 
Read_6/DisableCopyOnReadDisableCopyOnRead+read_6_disablecopyonread_output_nn_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp+read_6_disablecopyonread_output_nn_kernel_1^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

: }
Read_7/DisableCopyOnReadDisableCopyOnRead)read_7_disablecopyonread_output_nn_bias_1"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp)read_7_disablecopyonread_output_nn_bias_1^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_8/DisableCopyOnReadDisableCopyOnRead(read_8_disablecopyonread_dense_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp(read_8_disablecopyonread_dense_15_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_dense_15_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_dense_15_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_dense_16_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:@@|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_dense_16_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_dense_17_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@@|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_dense_17_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
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
:@
Read_14/DisableCopyOnReadDisableCopyOnRead*read_14_disablecopyonread_output_nn_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp*read_14_disablecopyonread_output_nn_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@}
Read_15/DisableCopyOnReadDisableCopyOnRead(read_15_disablecopyonread_output_nn_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp(read_15_disablecopyonread_output_nn_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_iteration^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_17/DisableCopyOnReadDisableCopyOnRead'read_17_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp'read_17_disablecopyonread_learning_rate^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead0read_18_disablecopyonread_adam_m_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp0read_18_disablecopyonread_adam_m_dense_12_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_19/DisableCopyOnReadDisableCopyOnRead0read_19_disablecopyonread_adam_v_dense_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp0read_19_disablecopyonread_adam_v_dense_12_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	� *
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	� f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	� �
Read_20/DisableCopyOnReadDisableCopyOnRead.read_20_disablecopyonread_adam_m_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp.read_20_disablecopyonread_adam_m_dense_12_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_adam_v_dense_12_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_adam_v_dense_12_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_m_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_m_dense_13_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_v_dense_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_v_dense_13_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_24/DisableCopyOnReadDisableCopyOnRead.read_24_disablecopyonread_adam_m_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp.read_24_disablecopyonread_adam_m_dense_13_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_adam_v_dense_13_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_adam_v_dense_13_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_14_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_14_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:  *
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:  e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:  �
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_m_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_m_dense_14_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
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
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_adam_v_dense_14_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_adam_v_dense_14_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
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
Read_30/DisableCopyOnReadDisableCopyOnRead3read_30_disablecopyonread_adam_m_output_nn_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp3read_30_disablecopyonread_adam_m_output_nn_kernel_1^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_31/DisableCopyOnReadDisableCopyOnRead3read_31_disablecopyonread_adam_v_output_nn_kernel_1"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp3read_31_disablecopyonread_adam_v_output_nn_kernel_1^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_32/DisableCopyOnReadDisableCopyOnRead1read_32_disablecopyonread_adam_m_output_nn_bias_1"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp1read_32_disablecopyonread_adam_m_output_nn_bias_1^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnRead1read_33_disablecopyonread_adam_v_output_nn_bias_1"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp1read_33_disablecopyonread_adam_v_output_nn_bias_1^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_dense_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_dense_15_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_dense_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_dense_15_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�@*
dtype0p
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�@f
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@�
Read_36/DisableCopyOnReadDisableCopyOnRead.read_36_disablecopyonread_adam_m_dense_15_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp.read_36_disablecopyonread_adam_m_dense_15_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_37/DisableCopyOnReadDisableCopyOnRead.read_37_disablecopyonread_adam_v_dense_15_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp.read_37_disablecopyonread_adam_v_dense_15_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_dense_16_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_dense_16_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_dense_16_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_40/DisableCopyOnReadDisableCopyOnRead.read_40_disablecopyonread_adam_m_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp.read_40_disablecopyonread_adam_m_dense_16_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_41/DisableCopyOnReadDisableCopyOnRead.read_41_disablecopyonread_adam_v_dense_16_bias"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp.read_41_disablecopyonread_adam_v_dense_16_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_dense_17_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_dense_17_kernel"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_dense_17_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_44/DisableCopyOnReadDisableCopyOnRead.read_44_disablecopyonread_adam_m_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp.read_44_disablecopyonread_adam_m_dense_17_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_45/DisableCopyOnReadDisableCopyOnRead.read_45_disablecopyonread_adam_v_dense_17_bias"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp.read_45_disablecopyonread_adam_v_dense_17_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_46/DisableCopyOnReadDisableCopyOnRead1read_46_disablecopyonread_adam_m_output_nn_kernel"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp1read_46_disablecopyonread_adam_m_output_nn_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_47/DisableCopyOnReadDisableCopyOnRead1read_47_disablecopyonread_adam_v_output_nn_kernel"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp1read_47_disablecopyonread_adam_v_output_nn_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_48/DisableCopyOnReadDisableCopyOnRead/read_48_disablecopyonread_adam_m_output_nn_bias"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp/read_48_disablecopyonread_adam_m_output_nn_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
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
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_adam_v_output_nn_bias"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_adam_v_output_nn_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
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
:t
Read_50/DisableCopyOnReadDisableCopyOnReadread_50_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOpread_50_disablecopyonread_total^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_51/DisableCopyOnReadDisableCopyOnReadread_51_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpread_51_disablecopyonread_count^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*�
value�B�5B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *C
dtypes9
725	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_104Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_105IdentityIdentity_104:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_105Identity_105:output:0*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_51/ReadVariableOpRead_51/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:5

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597027
dense_12_input#
dense_12_2597006:	� 
dense_12_2597008: "
dense_13_2597011:  
dense_13_2597013: "
dense_14_2597016:  
dense_14_2597018: #
output_nn_2597021: 
output_nn_2597023:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCalldense_12_inputdense_12_2597006dense_12_2597008*
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2596946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2597011dense_13_2597013*
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
E__inference_dense_13_layer_call_and_return_conditional_losses_2596963�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_2597016dense_14_2597018*
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2596980�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0output_nn_2597021output_nn_2597023*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2596996y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_12_input
�

�
E__inference_dense_17_layer_call_and_return_conditional_losses_2598491

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
E__inference_dense_17_layer_call_and_return_conditional_losses_2597248

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
F__inference_output_NN_layer_call_and_return_conditional_losses_2598432

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
E__inference_dense_13_layer_call_and_return_conditional_losses_2596963

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
�
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597054

inputs#
dense_12_2597033:	� 
dense_12_2597035: "
dense_13_2597038:  
dense_13_2597040: "
dense_14_2597043:  
dense_14_2597045: #
output_nn_2597048: 
output_nn_2597050:
identity�� dense_12/StatefulPartitionedCall� dense_13/StatefulPartitionedCall� dense_14/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCallinputsdense_12_2597033dense_12_2597035*
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2596946�
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2597038dense_13_2597040*
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
E__inference_dense_13_layer_call_and_return_conditional_losses_2596963�
 dense_14/StatefulPartitionedCallStatefulPartitionedCall)dense_13/StatefulPartitionedCall:output:0dense_14_2597043dense_14_2597045*
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2596980�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_14/StatefulPartitionedCall:output:0output_nn_2597048output_nn_2597050*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2596996y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
E__inference_model1_4_layer_call_and_return_conditional_losses_2597648

inputs
inputs_1#
group_nn_2597598:	� 
group_nn_2597600: "
group_nn_2597602:  
group_nn_2597604: "
group_nn_2597606:  
group_nn_2597608: "
group_nn_2597610: 
group_nn_2597612:'
technique_nn_2597615:	�@"
technique_nn_2597617:@&
technique_nn_2597619:@@"
technique_nn_2597621:@&
technique_nn_2597623:@@"
technique_nn_2597625:@&
technique_nn_2597627:@"
technique_nn_2597629:
identity�� Group_NN/StatefulPartitionedCall�$Technique_NN/StatefulPartitionedCall�
 Group_NN/StatefulPartitionedCallStatefulPartitionedCallinputsgroup_nn_2597598group_nn_2597600group_nn_2597602group_nn_2597604group_nn_2597606group_nn_2597608group_nn_2597610group_nn_2597612*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597054�
$Technique_NN/StatefulPartitionedCallStatefulPartitionedCallinputs_1technique_nn_2597615technique_nn_2597617technique_nn_2597619technique_nn_2597621technique_nn_2597623technique_nn_2597625technique_nn_2597627technique_nn_2597629*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597322z
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
dot_4/PartitionedCallPartitionedCalll2_normalize:z:0l2_normalize_1:z:0*
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2597533m
IdentityIdentitydot_4/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^Group_NN/StatefulPartitionedCall%^Technique_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2D
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
�
S
'__inference_dot_4_layer_call_fn_2598342
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2597533`
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
�
*__inference_Group_NN_layer_call_fn_2597118
dense_12_input
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597099o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_12_input
�	
�
*__inference_Group_NN_layer_call_fn_2597073
dense_12_input
unknown:	� 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597054o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_12_input
�#
�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598204

inputs:
'dense_12_matmul_readvariableop_resource:	� 6
(dense_12_biasadd_readvariableop_resource: 9
'dense_13_matmul_readvariableop_resource:  6
(dense_13_biasadd_readvariableop_resource: 9
'dense_14_matmul_readvariableop_resource:  6
(dense_14_biasadd_readvariableop_resource: :
(output_nn_matmul_readvariableop_resource: 7
)output_nn_biasadd_readvariableop_resource:
identity��dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�dense_13/BiasAdd/ReadVariableOp�dense_13/MatMul/ReadVariableOp�dense_14/BiasAdd/ReadVariableOp�dense_14/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0{
dense_12/MatMulMatMulinputs&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_13/MatMulMatMuldense_12/BiasAdd:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_13/ReluReludense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
dense_14/MatMul/ReadVariableOpReadVariableOp'dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
dense_14/MatMulMatMuldense_13/Relu:activations:0&dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
dense_14/BiasAddBiasAdddense_14/MatMul:product:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� b
dense_14/ReluReludense_14/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
output_NN/MatMulMatMuldense_14/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp^dense_14/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2@
dense_14/MatMul/ReadVariableOpdense_14/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�f
�
E__inference_model1_4_layer_call_and_return_conditional_losses_2598132
inputs_input_group
inputs_input_techniqueC
0group_nn_dense_12_matmul_readvariableop_resource:	� ?
1group_nn_dense_12_biasadd_readvariableop_resource: B
0group_nn_dense_13_matmul_readvariableop_resource:  ?
1group_nn_dense_13_biasadd_readvariableop_resource: B
0group_nn_dense_14_matmul_readvariableop_resource:  ?
1group_nn_dense_14_biasadd_readvariableop_resource: C
1group_nn_output_nn_matmul_readvariableop_resource: @
2group_nn_output_nn_biasadd_readvariableop_resource:G
4technique_nn_dense_15_matmul_readvariableop_resource:	�@C
5technique_nn_dense_15_biasadd_readvariableop_resource:@F
4technique_nn_dense_16_matmul_readvariableop_resource:@@C
5technique_nn_dense_16_biasadd_readvariableop_resource:@F
4technique_nn_dense_17_matmul_readvariableop_resource:@@C
5technique_nn_dense_17_biasadd_readvariableop_resource:@G
5technique_nn_output_nn_matmul_readvariableop_resource:@D
6technique_nn_output_nn_biasadd_readvariableop_resource:
identity��(Group_NN/dense_12/BiasAdd/ReadVariableOp�'Group_NN/dense_12/MatMul/ReadVariableOp�(Group_NN/dense_13/BiasAdd/ReadVariableOp�'Group_NN/dense_13/MatMul/ReadVariableOp�(Group_NN/dense_14/BiasAdd/ReadVariableOp�'Group_NN/dense_14/MatMul/ReadVariableOp�)Group_NN/output_NN/BiasAdd/ReadVariableOp�(Group_NN/output_NN/MatMul/ReadVariableOp�,Technique_NN/dense_15/BiasAdd/ReadVariableOp�+Technique_NN/dense_15/MatMul/ReadVariableOp�,Technique_NN/dense_16/BiasAdd/ReadVariableOp�+Technique_NN/dense_16/MatMul/ReadVariableOp�,Technique_NN/dense_17/BiasAdd/ReadVariableOp�+Technique_NN/dense_17/MatMul/ReadVariableOp�-Technique_NN/output_NN/BiasAdd/ReadVariableOp�,Technique_NN/output_NN/MatMul/ReadVariableOp�
'Group_NN/dense_12/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_12_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype0�
Group_NN/dense_12/MatMulMatMulinputs_input_group/Group_NN/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_12/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_12_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_12/BiasAddBiasAdd"Group_NN/dense_12/MatMul:product:00Group_NN/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_13/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_13_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_13/MatMulMatMul"Group_NN/dense_12/BiasAdd:output:0/Group_NN/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_13/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_13/BiasAddBiasAdd"Group_NN/dense_13/MatMul:product:00Group_NN/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_13/ReluRelu"Group_NN/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
'Group_NN/dense_14/MatMul/ReadVariableOpReadVariableOp0group_nn_dense_14_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0�
Group_NN/dense_14/MatMulMatMul$Group_NN/dense_13/Relu:activations:0/Group_NN/dense_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
(Group_NN/dense_14/BiasAdd/ReadVariableOpReadVariableOp1group_nn_dense_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Group_NN/dense_14/BiasAddBiasAdd"Group_NN/dense_14/MatMul:product:00Group_NN/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� t
Group_NN/dense_14/ReluRelu"Group_NN/dense_14/BiasAdd:output:0*
T0*'
_output_shapes
:��������� �
(Group_NN/output_NN/MatMul/ReadVariableOpReadVariableOp1group_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Group_NN/output_NN/MatMulMatMul$Group_NN/dense_14/Relu:activations:00Group_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
+Technique_NN/dense_15/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
Technique_NN/dense_15/MatMulMatMulinputs_input_technique3Technique_NN/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_15/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_15/BiasAddBiasAdd&Technique_NN/dense_15/MatMul:product:04Technique_NN/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_16/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_16/MatMulMatMul&Technique_NN/dense_15/BiasAdd:output:03Technique_NN/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_16/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_16/BiasAddBiasAdd&Technique_NN/dense_16/MatMul:product:04Technique_NN/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_16/ReluRelu&Technique_NN/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
+Technique_NN/dense_17/MatMul/ReadVariableOpReadVariableOp4technique_nn_dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
Technique_NN/dense_17/MatMulMatMul(Technique_NN/dense_16/Relu:activations:03Technique_NN/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/dense_17/BiasAdd/ReadVariableOpReadVariableOp5technique_nn_dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
Technique_NN/dense_17/BiasAddBiasAdd&Technique_NN/dense_17/MatMul:product:04Technique_NN/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
Technique_NN/dense_17/ReluRelu&Technique_NN/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
,Technique_NN/output_NN/MatMul/ReadVariableOpReadVariableOp5technique_nn_output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
Technique_NN/output_NN/MatMulMatMul(Technique_NN/dense_17/Relu:activations:04Technique_NN/output_NN/MatMul/ReadVariableOp:value:0*
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
dot_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_4/ExpandDims
ExpandDimsl2_normalize:z:0dot_4/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������X
dot_4/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot_4/ExpandDims_1
ExpandDimsl2_normalize_1:z:0dot_4/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:����������
dot_4/MatMulBatchMatMulV2dot_4/ExpandDims:output:0dot_4/ExpandDims_1:output:0*
T0*+
_output_shapes
:���������^
dot_4/ShapeShapedot_4/MatMul:output:0*
T0*
_output_shapes
::��x
dot_4/SqueezeSqueezedot_4/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims
e
IdentityIdentitydot_4/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^Group_NN/dense_12/BiasAdd/ReadVariableOp(^Group_NN/dense_12/MatMul/ReadVariableOp)^Group_NN/dense_13/BiasAdd/ReadVariableOp(^Group_NN/dense_13/MatMul/ReadVariableOp)^Group_NN/dense_14/BiasAdd/ReadVariableOp(^Group_NN/dense_14/MatMul/ReadVariableOp*^Group_NN/output_NN/BiasAdd/ReadVariableOp)^Group_NN/output_NN/MatMul/ReadVariableOp-^Technique_NN/dense_15/BiasAdd/ReadVariableOp,^Technique_NN/dense_15/MatMul/ReadVariableOp-^Technique_NN/dense_16/BiasAdd/ReadVariableOp,^Technique_NN/dense_16/MatMul/ReadVariableOp-^Technique_NN/dense_17/BiasAdd/ReadVariableOp,^Technique_NN/dense_17/MatMul/ReadVariableOp.^Technique_NN/output_NN/BiasAdd/ReadVariableOp-^Technique_NN/output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:����������:����������: : : : : : : : : : : : : : : : 2T
(Group_NN/dense_12/BiasAdd/ReadVariableOp(Group_NN/dense_12/BiasAdd/ReadVariableOp2R
'Group_NN/dense_12/MatMul/ReadVariableOp'Group_NN/dense_12/MatMul/ReadVariableOp2T
(Group_NN/dense_13/BiasAdd/ReadVariableOp(Group_NN/dense_13/BiasAdd/ReadVariableOp2R
'Group_NN/dense_13/MatMul/ReadVariableOp'Group_NN/dense_13/MatMul/ReadVariableOp2T
(Group_NN/dense_14/BiasAdd/ReadVariableOp(Group_NN/dense_14/BiasAdd/ReadVariableOp2R
'Group_NN/dense_14/MatMul/ReadVariableOp'Group_NN/dense_14/MatMul/ReadVariableOp2V
)Group_NN/output_NN/BiasAdd/ReadVariableOp)Group_NN/output_NN/BiasAdd/ReadVariableOp2T
(Group_NN/output_NN/MatMul/ReadVariableOp(Group_NN/output_NN/MatMul/ReadVariableOp2\
,Technique_NN/dense_15/BiasAdd/ReadVariableOp,Technique_NN/dense_15/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_15/MatMul/ReadVariableOp+Technique_NN/dense_15/MatMul/ReadVariableOp2\
,Technique_NN/dense_16/BiasAdd/ReadVariableOp,Technique_NN/dense_16/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_16/MatMul/ReadVariableOp+Technique_NN/dense_16/MatMul/ReadVariableOp2\
,Technique_NN/dense_17/BiasAdd/ReadVariableOp,Technique_NN/dense_17/BiasAdd/ReadVariableOp2Z
+Technique_NN/dense_17/MatMul/ReadVariableOp+Technique_NN/dense_17/MatMul/ReadVariableOp2^
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
�
�
*__inference_dense_13_layer_call_fn_2598382

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
E__inference_dense_13_layer_call_and_return_conditional_losses_2596963o
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2598451

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
n
B__inference_dot_4_layer_call_and_return_conditional_losses_2598354
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
�#
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598306

inputs:
'dense_15_matmul_readvariableop_resource:	�@6
(dense_15_biasadd_readvariableop_resource:@9
'dense_16_matmul_readvariableop_resource:@@6
(dense_16_biasadd_readvariableop_resource:@9
'dense_17_matmul_readvariableop_resource:@@6
(dense_17_biasadd_readvariableop_resource:@:
(output_nn_matmul_readvariableop_resource:@7
)output_nn_biasadd_readvariableop_resource:
identity��dense_15/BiasAdd/ReadVariableOp�dense_15/MatMul/ReadVariableOp�dense_16/BiasAdd/ReadVariableOp�dense_16/MatMul/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�dense_17/MatMul/ReadVariableOp� output_NN/BiasAdd/ReadVariableOp�output_NN/MatMul/ReadVariableOp�
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0{
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_16/MatMulMatMuldense_15/BiasAdd:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@b
dense_17/ReluReludense_17/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
output_NN/MatMul/ReadVariableOpReadVariableOp(output_nn_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
output_NN/MatMulMatMuldense_17/Relu:activations:0'output_NN/MatMul/ReadVariableOp:value:0*
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
:����������
NoOpNoOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp!^output_NN/BiasAdd/ReadVariableOp ^output_NN/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp2D
 output_NN/BiasAdd/ReadVariableOp output_NN/BiasAdd/ReadVariableOp2B
output_NN/MatMul/ReadVariableOpoutput_NN/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597271
dense_15_input#
dense_15_2597215:	�@
dense_15_2597217:@"
dense_16_2597232:@@
dense_16_2597234:@"
dense_17_2597249:@@
dense_17_2597251:@#
output_nn_2597265:@
output_nn_2597267:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCalldense_15_inputdense_15_2597215dense_15_2597217*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2597214�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_2597232dense_16_2597234*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2597231�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_2597249dense_17_2597251*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_2597248�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0output_nn_2597265output_nn_2597267*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2597264y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_15_input
�
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597295
dense_15_input#
dense_15_2597274:	�@
dense_15_2597276:@"
dense_16_2597279:@@
dense_16_2597281:@"
dense_17_2597284:@@
dense_17_2597286:@#
output_nn_2597289:@
output_nn_2597291:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCalldense_15_inputdense_15_2597274dense_15_2597276*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2597214�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_2597279dense_16_2597281*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2597231�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_2597284dense_17_2597286*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_2597248�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0output_nn_2597289output_nn_2597291*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2597264y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_15_input
�
�
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597367

inputs#
dense_15_2597346:	�@
dense_15_2597348:@"
dense_16_2597351:@@
dense_16_2597353:@"
dense_17_2597356:@@
dense_17_2597358:@#
output_nn_2597361:@
output_nn_2597363:
identity�� dense_15/StatefulPartitionedCall� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�!output_NN/StatefulPartitionedCall�
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_2597346dense_15_2597348*
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2597214�
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_2597351dense_16_2597353*
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2597231�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_2597356dense_17_2597358*
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
E__inference_dense_17_layer_call_and_return_conditional_losses_2597248�
!output_NN/StatefulPartitionedCallStatefulPartitionedCall)dense_17/StatefulPartitionedCall:output:0output_nn_2597361output_nn_2597363*
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2597264y
IdentityIdentity*output_NN/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^output_NN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:����������: : : : : : : : 2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!output_NN/StatefulPartitionedCall!output_NN/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_output_NN_layer_call_fn_2598422

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
F__inference_output_NN_layer_call_and_return_conditional_losses_2596996o
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
 
_user_specified_nameinputs"�
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
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
15"
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
15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
 layer_regularization_losses
!layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
"trace_0
#trace_1
$trace_2
%trace_32�
*__inference_model1_4_layer_call_fn_2597683
*__inference_model1_4_layer_call_fn_2597775
*__inference_model1_4_layer_call_fn_2597938
*__inference_model1_4_layer_call_fn_2597976�
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
 z"trace_0z#trace_1z$trace_2z%trace_3
�
&trace_0
'trace_1
(trace_2
)trace_32�
E__inference_model1_4_layer_call_and_return_conditional_losses_2597536
E__inference_model1_4_layer_call_and_return_conditional_losses_2597590
E__inference_model1_4_layer_call_and_return_conditional_losses_2598054
E__inference_model1_4_layer_call_and_return_conditional_losses_2598132�
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
 z&trace_0z'trace_1z(trace_2z)trace_3
�B�
"__inference__wrapped_model_2596932input_Groupinput_Technique"�
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
�
*layer_with_weights-0
*layer-0
+layer_with_weights-1
+layer-1
,layer_with_weights-2
,layer-2
-layer_with_weights-3
-layer-3
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
4layer_with_weights-0
4layer-0
5layer_with_weights-1
5layer-1
6layer_with_weights-2
6layer-2
7layer_with_weights-3
7layer-3
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
�
D
_variables
E_iterations
F_learning_rate
G_index_dict
H
_momentums
I_velocities
J_update_step_xla"
experimentalOptimizer
,
Kserving_default"
signature_map
": 	� 2dense_12/kernel
: 2dense_12/bias
!:  2dense_13/kernel
: 2dense_13/bias
!:  2dense_14/kernel
: 2dense_14/bias
":  2output_NN/kernel
:2output_NN/bias
": 	�@2dense_15/kernel
:@2dense_15/bias
!:@@2dense_16/kernel
:@2dense_16/bias
!:@@2dense_17/kernel
:@2dense_17/bias
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
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_model1_4_layer_call_fn_2597683input_Groupinput_Technique"�
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
*__inference_model1_4_layer_call_fn_2597775input_Groupinput_Technique"�
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
*__inference_model1_4_layer_call_fn_2597938inputs_input_groupinputs_input_technique"�
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
*__inference_model1_4_layer_call_fn_2597976inputs_input_groupinputs_input_technique"�
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2597536input_Groupinput_Technique"�
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2597590input_Groupinput_Technique"�
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2598054inputs_input_groupinputs_input_technique"�
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2598132inputs_input_groupinputs_input_technique"�
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
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
jtrace_0
ktrace_1
ltrace_2
mtrace_32�
*__inference_Group_NN_layer_call_fn_2597073
*__inference_Group_NN_layer_call_fn_2597118
*__inference_Group_NN_layer_call_fn_2598153
*__inference_Group_NN_layer_call_fn_2598174�
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
 zjtrace_0zktrace_1zltrace_2zmtrace_3
�
ntrace_0
otrace_1
ptrace_2
qtrace_32�
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597003
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597027
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598204
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598234�
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
 zntrace_0zotrace_1zptrace_2zqtrace_3
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
~	variables
trainable_variables
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
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
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
.__inference_Technique_NN_layer_call_fn_2597341
.__inference_Technique_NN_layer_call_fn_2597386
.__inference_Technique_NN_layer_call_fn_2598255
.__inference_Technique_NN_layer_call_fn_2598276�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597271
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597295
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598306
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598336�
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
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dot_4_layer_call_fn_2598342�
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2598354�
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
E0
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
�32"
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
�15"
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
�15"
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
%__inference_signature_wrapper_2597900input_Groupinput_Technique"�
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
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_12_layer_call_fn_2598363�
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2598373�
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
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_13_layer_call_fn_2598382�
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
E__inference_dense_13_layer_call_and_return_conditional_losses_2598393�
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
*__inference_dense_14_layer_call_fn_2598402�
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2598413�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_output_NN_layer_call_fn_2598422�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2598432�
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
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_Group_NN_layer_call_fn_2597073dense_12_input"�
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
*__inference_Group_NN_layer_call_fn_2597118dense_12_input"�
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
*__inference_Group_NN_layer_call_fn_2598153inputs"�
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
*__inference_Group_NN_layer_call_fn_2598174inputs"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597003dense_12_input"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597027dense_12_input"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598204inputs"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598234inputs"�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_15_layer_call_fn_2598441�
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2598451�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_16_layer_call_fn_2598460�
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2598471�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_17_layer_call_fn_2598480�
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
E__inference_dense_17_layer_call_and_return_conditional_losses_2598491�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_output_NN_layer_call_fn_2598500�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2598510�
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
 "
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_Technique_NN_layer_call_fn_2597341dense_15_input"�
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
.__inference_Technique_NN_layer_call_fn_2597386dense_15_input"�
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
.__inference_Technique_NN_layer_call_fn_2598255inputs"�
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
.__inference_Technique_NN_layer_call_fn_2598276inputs"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597271dense_15_input"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597295dense_15_input"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598306inputs"�
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598336inputs"�
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
'__inference_dot_4_layer_call_fn_2598342inputs_0inputs_1"�
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
B__inference_dot_4_layer_call_and_return_conditional_losses_2598354inputs_0inputs_1"�
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
':%	� 2Adam/m/dense_12/kernel
':%	� 2Adam/v/dense_12/kernel
 : 2Adam/m/dense_12/bias
 : 2Adam/v/dense_12/bias
&:$  2Adam/m/dense_13/kernel
&:$  2Adam/v/dense_13/kernel
 : 2Adam/m/dense_13/bias
 : 2Adam/v/dense_13/bias
&:$  2Adam/m/dense_14/kernel
&:$  2Adam/v/dense_14/kernel
 : 2Adam/m/dense_14/bias
 : 2Adam/v/dense_14/bias
':% 2Adam/m/output_NN/kernel
':% 2Adam/v/output_NN/kernel
!:2Adam/m/output_NN/bias
!:2Adam/v/output_NN/bias
':%	�@2Adam/m/dense_15/kernel
':%	�@2Adam/v/dense_15/kernel
 :@2Adam/m/dense_15/bias
 :@2Adam/v/dense_15/bias
&:$@@2Adam/m/dense_16/kernel
&:$@@2Adam/v/dense_16/kernel
 :@2Adam/m/dense_16/bias
 :@2Adam/v/dense_16/bias
&:$@@2Adam/m/dense_17/kernel
&:$@@2Adam/v/dense_17/kernel
 :@2Adam/m/dense_17/bias
 :@2Adam/v/dense_17/bias
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
*__inference_dense_12_layer_call_fn_2598363inputs"�
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2598373inputs"�
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
*__inference_dense_13_layer_call_fn_2598382inputs"�
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
E__inference_dense_13_layer_call_and_return_conditional_losses_2598393inputs"�
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
*__inference_dense_14_layer_call_fn_2598402inputs"�
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
E__inference_dense_14_layer_call_and_return_conditional_losses_2598413inputs"�
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
+__inference_output_NN_layer_call_fn_2598422inputs"�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2598432inputs"�
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
*__inference_dense_15_layer_call_fn_2598441inputs"�
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
E__inference_dense_15_layer_call_and_return_conditional_losses_2598451inputs"�
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
*__inference_dense_16_layer_call_fn_2598460inputs"�
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
E__inference_dense_16_layer_call_and_return_conditional_losses_2598471inputs"�
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
*__inference_dense_17_layer_call_fn_2598480inputs"�
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
E__inference_dense_17_layer_call_and_return_conditional_losses_2598491inputs"�
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
+__inference_output_NN_layer_call_fn_2598500inputs"�
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2598510inputs"�
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597003z@�=
6�3
)�&
dense_12_input����������
p

 
� ",�)
"�
tensor_0���������
� �
E__inference_Group_NN_layer_call_and_return_conditional_losses_2597027z@�=
6�3
)�&
dense_12_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598204r8�5
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
E__inference_Group_NN_layer_call_and_return_conditional_losses_2598234r8�5
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
*__inference_Group_NN_layer_call_fn_2597073o@�=
6�3
)�&
dense_12_input����������
p

 
� "!�
unknown����������
*__inference_Group_NN_layer_call_fn_2597118o@�=
6�3
)�&
dense_12_input����������
p 

 
� "!�
unknown����������
*__inference_Group_NN_layer_call_fn_2598153g8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
*__inference_Group_NN_layer_call_fn_2598174g8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597271z@�=
6�3
)�&
dense_15_input����������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2597295z@�=
6�3
)�&
dense_15_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598306r8�5
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
I__inference_Technique_NN_layer_call_and_return_conditional_losses_2598336r8�5
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
.__inference_Technique_NN_layer_call_fn_2597341o@�=
6�3
)�&
dense_15_input����������
p

 
� "!�
unknown����������
.__inference_Technique_NN_layer_call_fn_2597386o@�=
6�3
)�&
dense_15_input����������
p 

 
� "!�
unknown����������
.__inference_Technique_NN_layer_call_fn_2598255g8�5
.�+
!�
inputs����������
p

 
� "!�
unknown����������
.__inference_Technique_NN_layer_call_fn_2598276g8�5
.�+
!�
inputs����������
p 

 
� "!�
unknown����������
"__inference__wrapped_model_2596932����
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
E__inference_dense_12_layer_call_and_return_conditional_losses_2598373d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_12_layer_call_fn_2598363Y0�-
&�#
!�
inputs����������
� "!�
unknown��������� �
E__inference_dense_13_layer_call_and_return_conditional_losses_2598393c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_13_layer_call_fn_2598382X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_dense_14_layer_call_and_return_conditional_losses_2598413c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0��������� 
� �
*__inference_dense_14_layer_call_fn_2598402X/�,
%�"
 �
inputs��������� 
� "!�
unknown��������� �
E__inference_dense_15_layer_call_and_return_conditional_losses_2598451d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_15_layer_call_fn_2598441Y0�-
&�#
!�
inputs����������
� "!�
unknown���������@�
E__inference_dense_16_layer_call_and_return_conditional_losses_2598471c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_16_layer_call_fn_2598460X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
E__inference_dense_17_layer_call_and_return_conditional_losses_2598491c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_dense_17_layer_call_fn_2598480X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
B__inference_dot_4_layer_call_and_return_conditional_losses_2598354�Z�W
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
'__inference_dot_4_layer_call_fn_2598342Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� "!�
unknown����������
E__inference_model1_4_layer_call_and_return_conditional_losses_2597536����
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2597590����
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2598054����
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
E__inference_model1_4_layer_call_and_return_conditional_losses_2598132����
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
*__inference_model1_4_layer_call_fn_2597683����
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
*__inference_model1_4_layer_call_fn_2597775����
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
*__inference_model1_4_layer_call_fn_2597938����
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
*__inference_model1_4_layer_call_fn_2597976����
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
F__inference_output_NN_layer_call_and_return_conditional_losses_2598432c/�,
%�"
 �
inputs��������� 
� ",�)
"�
tensor_0���������
� �
F__inference_output_NN_layer_call_and_return_conditional_losses_2598510c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
+__inference_output_NN_layer_call_fn_2598422X/�,
%�"
 �
inputs��������� 
� "!�
unknown����������
+__inference_output_NN_layer_call_fn_2598500X/�,
%�"
 �
inputs���������@
� "!�
unknown����������
%__inference_signature_wrapper_2597900����
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