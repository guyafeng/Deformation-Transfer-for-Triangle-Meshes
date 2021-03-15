### Reproduction: Deformation Transfer for Triangle Meshes
This is a Python implementation of the paper "Deformation Transfer for Triangle Meshes".
I think this repo would be more in-line with the methodology which demonstrated in the paper.
Since I didn't optimize the result in an iterative way, instead I use LU factorization to obtain the 
Moore-Penrose pseudo inverse then calculate the optimal solution of the linear optimization problem.

For Non-rigid registration part, I derive the constraint optimization problem with Lagrangian-Multiplier, 
then convert it to a un-constraint linear optimization, and solving the problem with LU factorization
as well.
#### Request Packages
```python
numpy 1.18.0
scipy 1.4.1
```
In this project, the sparse matrix calculation module of scipy is the key tool for solving the linear 
optimization problem.

#### Demo
##### Results of Non-rigid registration
![non-rigid registration](https://images.algorithmic.cn/GitHub/images/deformation_transfer_demo1.png)

keep updating...