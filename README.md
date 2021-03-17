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

##### Results of Deformation transfer
![deformation transfer](https://images.algorithmic.cn/GitHub/images/deformation_transfer_demo2.png)
The topology of source & target mesh is different with each other.


##### Time consuming
- non-rigid registration stage: about 20 minutes! Though it's pretty slow, we only need to calculate it for once if we
have a new pair of source/target mesh.

- correspondence finding: about 1 minute.

- deformation transfer: few seconds.

The speed of LU factorization seems has inversely proportional relationship with the number of vertices of mesh.
So, use mesh has lower vertices and triangle faces might be faster.

#### Things that REALLY should be NOTICED!
The [obj data file released by the author Dr. R.W. Sumner](http://people.csail.mit.edu/sumner/research/deftransfer/data.html)
has some mistakes in "face-poses". The order of vertices of reference mesh and some deformed mesh are different!!!
e.g. face-01-anger.obj, face-03-fury.obj. As far as I know, only the order of face-02-cry.obj has the same order of
vertices as face-reference.obj. So only transfer the deformation of face-02-cry.obj could obtain a correct result while
using my code.

![wrong vertices order](https://images.algorithmic.cn/GitHub/images/deformation_transfer_wrong_vts_order.png)

I'm think the order of face-reference.obj and other deformed face-xx-xxx.obj might be the same, follow this you could still obtain
correct deformation transfer results if made some modification based on my code.

In my other tests, if the vertices order of reference and deformed source mesh can be guaranteed, the results are always
good!


#### Reference
- [1] [Deformation Transfer for Triangle Meshes: R.W. Sumner etal.](https://dl.acm.org/doi/abs/10.1145/1015706.1015736)
- [2] [Mesh Modification Using Deformation Gradients: R.W. Sumner](https://dspace.mit.edu/handle/1721.1/34025)