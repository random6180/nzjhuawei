华为软件精英挑战赛
====
一．决策树设计
---
1.树节点设计
```cpp
class treenode {
public:
	treenode() : col(-1), value(0), left(nullptr), right(nullptr) {}
	treenode(int col, double value, vector<double> result) :
		col(col), value(value), left(nullptr), right(nullptr), result(result) { }
public:
	int col;		//以col作为特征划分
	double value;		//以小于等于value作为划分依据，满足则为左子树，否则为右子树
	treenode* left;
	treenode* right;
	vector<double> result; //只对叶节点存放输出值
};
```
说明：
节点的初始化，col代表特征对应的维度，value作为该特征的切分点，向量result存放叶节点的输出值集合，对于非叶节点，向量result设为空。

2.集合的切分
pair<vector<vector<double> >, vector<vector<double> >> divideTwo(vector<vector<double> >& inputvector, int col, double value) {	//以col和value将输入分为两部分
	vector<vector<double> > leftvector;
	vector<vector<double> > rightvector;
	for (unsigned int i = 0; i < inputvector.size(); i++) {
		if (inputvector[i][col] <= value)
			leftvector.push_back(inputvector[i]);
		else
			rightvector.push_back(inputvector[i]);
	}
	pair<vector<vector<double> >, vector<vector<double> >> a(leftvector, rightvector);
	return a;
}

说明：
inputvector是一个输入矩阵，每行代表一组输入向量和一个输出值。
该函数选取切分特征col，切分值value，把inputvector分成leftvector和rightvector两部分。

3.原始数据集到上面inputvector的转换

vector<vector<double> > divideserial(vector<double>& flavor, int num) {//输入向量为num+1维的，其中最后一维为输出
	vector<double> input(num + 1, 0);
	vector<vector<double> > inputvector;
	for (unsigned int i = 0; i < flavor.size() - num; i++) {
		for (int j = 0; j < num + 1; j++) {
			input[j] = flavor[i + j];
		}
		inputvector.push_back(input);
	}
	return inputvector;
}

说明：
flavor为虚拟机每天的数量，将其以num为维度进行切分，输入为前num个值，输出为后一个值。之所以这样切分是因为虚拟机的数量受前一段时间的影响。一般num选自然周期7。

4.生成树（核心）
treenode* buildTree(vector<vector<double> >& inputvector) {							//可能会超时
	treenode *node = new treenode();
	double minsquaresum = squareSum(inputvector);
	if (inputvector.size() <= 1 || minsquaresum == 0.0) {								//叶结点,只有一个向量，或者输出相同
		for (unsigned int i = 0; i < inputvector.size(); i++)
			(node->result).push_back(inputvector[i][inputvector[0].size() - 1]);
		return node;
	}

	pair<vector<vector<double> >, vector<vector<double> >> a;
	vector<vector<double> > leftvector;
	vector<vector<double> > rightvector;

	vector<vector<double> > finalleftvector;
	vector<vector<double> > finalrightvector;
	double squaresum = 0.0;

	unsigned int thecol = -1;
	double value = 0;
	bool flag = false;				//正常情况下flag为true

	for (unsigned col = 0; col < inputvector[0].size() - 1; col++) {
		for (unsigned j = 0; j < inputvector.size(); j++) {
			a = divideTwo(inputvector, col, inputvector[j][col]);		//切分
			leftvector = a.first;
			rightvector = a.second;
			squaresum = squareSum(leftvector) + squareSum(rightvector);	//存在无法分开的情形
			if (squaresum < minsquaresum) {
				flag = true;
				minsquaresum = squaresum;
				thecol = col;
				value = inputvector[j][col];
				finalleftvector = leftvector;
				finalrightvector = rightvector;
			}
		}
	}

	if (flag == false) {								//叶结点,只有一个向量，或者输出相同
		for (unsigned int i = 0; i < inputvector.size(); i++)
			(node->result).push_back(inputvector[i][inputvector[0].size() - 1]);
		return node;
	}
	node->col = thecol;
	node->value = value;
	//	treenode left(buildTree(finalleftvector));
	// 	treenode right(buildTree(finalrightvector));
	node->left = buildTree(finalleftvector);
	node->right = buildTree(finalrightvector);

	return node;
}

说明：
首先在栈上分配树节点，treenode *node = new treenode()；
squareSum(inputvector)计算输入样本的方差，判断方差为0（输出值相同）或只含一个样本，这种情况下为递归结束条件，直接作为叶节点。
否则，col遍历输入向量的特征，value遍历所有样本在col维度的值，并尝试以（col，value）切分，找出切分后leftvector和rightvector方差最小的组合作为最终切分点。最终以col，value初始化此非叶节点。
上述情形得到leftvector和rightvector又作为新的样本，在上面递归切分，
同时node->left = buildTree(finalleftvector)，node->right = buildTree(finalrightvector)，使node的左右孩子指针指向它们。

5.剪枝
void prune(treenode* node, double mingain = 2) {
	if (node->col == -1)
		return;
	if (node->left->col != -1)			//分支结点
		prune(node->left, mingain);
	if (node->right->col != -1)			//分支结点
		prune(node->right, mingain);

	if ((node->left->col == -1) && (node->right->col == -1)) {
		vector<double> mergeresult;
		double leftsquarediff = squareDiff(node->left->result);
		double rightsquarediff = squareDiff(node->right->result);
		for (unsigned int i = 0; i < node->left->result.size(); i++) {
			mergeresult.push_back(node->left->result[i]);
		}
		for (unsigned int i = 0; i < node->right->result.size(); i++) {
			mergeresult.push_back(node->right->result[i]);
		}
		double mergesquarediff = squareDiff(mergeresult);
		if (mergesquarediff - leftsquarediff - rightsquarediff < mingain) {		//方差减小显著
			delete node->left;
			delete node->right;
			node->left = nullptr;
			node->right = nullptr;
			node->col = -1;
			node->value = 0;
			node->result = mergeresult;
		}
	}
}

说明：
设定阈值mingain，当把原集合分割成两部分后，若方差减少量少于mingain，就对两个节点进行合并，这样自下而上剪枝。

6.打印树
void drawTree(treenode& node, string space = " ") {
	if (node.col != -1) {			//分支节点
		cout << "(" << node.col << "," << node.value << ")" << endl;
		cout << space;
		cout << "left:";
		drawTree(*node.left, space + ' ');1
		cout << space;
		cout << "right:";
		drawTree(*node.right, space + ' ');

	}
	else {
		cout << "leaf: ";
		for (unsigned int i = 0; i < node.result.size(); i++) {
			cout << node.result[i] << ' ';
		}
		cout << endl;
	}
}

说明：
如果是叶节点，则col=-1，打印存放的结果集合result；若是非叶节点，则col!=-1，打印切分点
（col，value），并递归打印左右孩子节点。


二．由决策树生成梯度提升树（gbdt）
---
1.生成提升树
vector<treenode*> gbdt(vector<vector<float> >& inputvector, int iter, float rate=1) {
	vector<treenode*> nodevector;
	treenode* node = new treenode;		//node为数组
	vector<float> results;
	vector<float> colvalues;
	vector<float> sortvalues;
	float sum2 = 0.0;
	float min = 10000000;
	float f0;

	colvalues = getColValues(inputvector, inputvector[0].size() - 1);		//即y值
	sortvalues = divideValue(colvalues);					//均匀分割数组	
	for (unsigned int i = 0; i < sortvalues.size(); i++) {
		for (unsigned int j = 0; j < colvalues.size(); j++)
			sum2 += (colvalues[j] - sortvalues[i])*(colvalues[j] - sortvalues[i]);
		if (sum2 < min) {
			min = sum2;
			f0 = sortvalues[i];
		}
		sum2 = 0;
	}

	node->result.push_back(f0);
	nodevector.push_back(node);

	vector<vector<float> > inputs;
	inputs = getInputValues(inputvector);
	float temp = 0.0;


	for (int k = 0; k < iter; k++) {
//		cout << "迭代轮数:" << k << endl;
		for (unsigned int t = 0; t < inputvector.size(); t++) {
			temp = gbdtPreOutput(nodevector, inputs[t]);
			inputvector[t][inputvector[0].size() - 1] = colvalues[t] - temp;	//计算残差
		}
		vector<float> tempcolvalues = getColValues(inputvector, inputvector[0].size() - 1);
		float sumdif = squareDiff(tempcolvalues);
//		cout << "方差:" << sumdif << endl;
		node = buildTree(inputvector);
		prune(node, sumdif*rate);
		nodevector.push_back(node);
	}

	return nodevector;
}

说明：
首先定义向量vector<treenode*> nodevector，内部存放各棵决策树头节点指针。
对于第一棵树，把它初始化为单节点，其值f0为，使所有输出值与f0差值平方和最小。
接下来进入k轮迭代，每次都根据之前k-1轮建立的决策树输出结果，并计算残差，继续拟合下一棵决策树，最终形成梯度提升树。

2.由提升树预测输出值
float gbdtPreOutput(vector<treenode*> nodevector, vector<float>& input) {
	float sum = 0.0;
	for (unsigned int i = 0; i < nodevector.size(); i++) {
		//		cout << "gbdt决策树" << i << ":" << preOutput(nodevector[i], input) << endl;
		sum += preOutput(nodevector[i], input);

	}
	return sum;
}

说明：
遍历vector<treenode*> nodevector，对每棵决策树，计算输出，并把结果累加，得到最终输出。



三．由决策树生成随机森林（rf）
---
vector<treenode*> randomForest(vector<vector<float> >& inputvector, int iter, float feature=0.6) {
	vector<treenode*> nodevector;
	treenode* node = new treenode;		//node为数组
	vector<float> results;
	vector<float> colvalues;
	vector<float> sortvalues;

	int sample = inputvector.size();			//取63.2%的样本


	for (int i = 0; i < iter; i++) {
		vector<int> rowset;							//存放样本索引
		srand((unsigned)time(NULL));
		int n = sample;
		while (n > 0) {
			int rd = rand() % inputvector.size();
				rowset.push_back(rd);
				n--;
		}
		vector<vector<float> > sampleinput;
		for (unsigned int j = 0; j < rowset.size(); j++) {
			sampleinput.push_back(inputvector[rowset[j]]);
		}
		node = buildTree(sampleinput, feature);
		nodevector.push_back(node);
		rowset.clear();
		sampleinput.clear();
	}
	return nodevector;
}

说明：
随机森林的实现比较简单，在原来决策树的基础上，随机取样并放回，并依次生成决策树。这样迭代iter轮，构成随机森林vector<treenode*> nodevector。输出值将取所有决策树的均值。
此处的buildTree函数经过修改，每棵树的特征选取随机化，本实现随机选取60%的特征。







四．局部加权线性回归
---
double lwlr(vector<double>& testPoint, vector<vector<double> >& xArr,
	vector<vector<double> >& yArr, double k = 1.0) {
	double result;
	vector<vector<double> > input;
	input.push_back(testPoint);
	Matrix point(input);

	Matrix xMat(xArr);
	Matrix yMat(yArr);
	int m = xArr.size();
	vector<vector<double> > weights;
	vector<double> temp;
	for (int i = 0; i < m; i++) {		//求出weights矩阵
		double w = 0;
		w = diffSquareSum(testPoint, xArr[i]);
//		w = 100 / (w + 1);
		w = exp(w / (-2 * k*k));// + 0.5*cos(0.8*sqrt(w));
		
		//		cout << "衰减系数w:" << w << endl;
		for (int j = 0; j < m; j++) {
			if (i == j)
				temp.push_back(w);
			else
				temp.push_back(0);
		}
		weights.push_back(temp);
		temp.clear();
	}

	Matrix W(weights);
	Matrix xTx;
	xTx = xMat.Transpose()*W*xMat;
	Matrix theta;
	Matrix pre;

	if (xTx.det() == 0)
		cout << "det = 0!" << endl;		//自身会抛出异常
	else
		theta = xTx.Inverse() * (xMat.Transpose() * (W * yMat));
	pre = point * theta;
	result = pre(1, 1);
	return result;
}
说明：
xArr是m*2矩阵，yArr是m*1矩阵，其中m为样本数，其中xArr矩阵第一列全部填充1。
testPoint为输入向量，其第一列也填充1，k为衰减系数。
接下来求出xArr矩阵中每行与testPoint的距离，与k一起构成权重矩阵weights。
最后借助矩阵运算库求出系数矩阵theta，由theta与point的乘积得到输出值。


五．局部加权非线性回归
---
1.得到系数theta
vector<double> generatetheta(vector<double>& xArr, vector<double>& yArr, double xp, int order = 10, int kp = 5) {//x为预测点
	vector<double> rowx;
	vector<vector<double> > x;
	vector<double> y;
	vector<vector<double> > ymat;
	double w;

	for (int i = 0; i < order + 1; i++) {
		for (int j = 0; j < order + 1; j++) {
			double xtemp1 = 0;
			for (unsigned int n = 0; n < xArr.size(); n++) {
				double xtemp2 = 1;
				w = exp((xp - xArr[n])*(xp - xArr[n]) / (-2 * kp * kp));
				for (int k = 0; k < i + j; k++) {
					xtemp2 *= xArr[n];
				}
				xtemp1 += xtemp2*w;
			}
			rowx.push_back(xtemp1);
		}
		x.push_back(rowx);
		rowx.clear();
	}

	for (int i = 0; i < order + 1; i++) {
		double ytemp1 = 0;
		for (unsigned int n = 0; n < yArr.size(); n++) {
			double ytemp2 = 1;
			w = exp((xp - xArr[n])*(xp - xArr[n]) / (-2 * kp*kp));
			//		w = 1;
			for (int k = 0; k < i; k++) {
				ytemp2 *= xArr[n];
			}
			ytemp2 *= yArr[n];
			ytemp2 *= w;
			ytemp1 += ytemp2;
		}
		y.push_back(ytemp1);
		ymat.push_back(y);
		y.clear();
	}

	Matrix X(x);
	Matrix Y(ymat);
	Matrix A;
	Matrix B;
	B = X.Inverse();

	B *= Y;
//	A = B * Y;
	A = B;
	vector<double> theta;
	for (int t = 0; t < A.getRow(); t++) {
		theta.push_back(A(t + 1, 1));
	}
	return theta;
}

说明：
局部加权非线性回归比lwlr多了个参数order，order为阶数。
上述代码两部分循环，第一个部分得到等式左边的矩阵，第二个循环得到等式右边的矩阵，满足
X*theta=Y，求出theta。

2.由系数theta预测输出
double preOutput(vector<double>& theta, double days) {
	/*double days = 1 / (1 + exp(-day));*/
	double result = 0;
	for (unsigned int i = 0; i < theta.size(); i++) {
		double temp1 = 1;
		for (unsigned int k = 0; k < i; k++) {
			temp1 *= days;
		}
		temp1 *= theta[i];
		result += temp1;
	}
	return result;
}

说明：
计算，变量k控制指数相乘次数，i控制系数个数。

六．虚拟机分配部分
---
1.贪心算法选出合适的虚拟机
int selectFlavor(vector<Flavor>& flavorlist, Server& server, int last, int first) {    //选出合适的k
	int minmum = 50000;
	int maxmum = 100;
	int minmum2 = 50000;
	int k = last;

	if (last < first)
		return -1;

	for (int i = last; i >= first; i--) {		//寻找k
		if (flavorlist[i].num > 0) {
			if (server.cpu == Cpu || isCpu * (int)((Memory - server.memory) / (Cpu - server.cpu)) < isCpu*(int)(Memory / Cpu)) {		//最不平衡
				int tmpmax = (int)(flavorlist[i].memory / flavorlist[i].cpu);
				if (isCpu == 1) {
					if (tmpmax > maxmum) {
						k = i;
						maxmum = tmpmax;
					}
				}
				else {
					if (tmpmax < minmum2) {
						k = i;
						minmum2 = tmpmax;
					}
				}
			}
			else {						//最平衡
				int tmpmin = simCalculate(flavorlist[i], server);
				if (tmpmin < minmum) {
					k = i;
					minmum = tmpmin;
				}
			}
		}
		else
			continue;
	}

	if (flavorlist[k].num > 0 && (server.cpu - flavorlist[k].cpu) >= 0 && (server.memory - flavorlist[k].memory >= 0))
		return k;
	else
		return selectFlavor(flavorlist, server, k - 1, first);		//递归寻找
}

说明：
flavorlist为各种规格虚拟机的数量，server为当前物理机的内存、cpu使用情况，last与first为flavorlist的索引区间。
此函数根据server物理机的信息，遍历flavorlist的first至last区间，从中选出合适的虚拟机分配到物理机上。
具体的先对虚拟机按规格排序，从大到小选取虚拟机，并计算物理机内存与cpu的原始比例。根据物理机剩余内存与cpu比值，作出以下两种选择。若比值低于/高于原始比例，则选取虚拟机比值最大/最小的放入；否则，每次选取虚拟机，使物理机剩余比例尽量接近原始比例。
若选出这样的虚拟机索引k，且不溢出，则返回；否则，递归调用函数，改变函数参数last为k-1。
若全部遍历完，仍溢出，则返回-1。

2.分配方案
int allocate(vector<Flavor>& flavorlist, vector<Server>& serverlist) {
	int sumflavor = 0;										//虚拟机总数
	int k = 0;												//虚拟机索引号

	for (unsigned int i = 0; i < flavorlist.size(); i++) {
		sumflavor += flavorlist[i].num;
	}

	serverlist.push_back(Server());
	while (sumflavor > 0) {
		k = selectFlavor(flavorlist, serverlist[Server::counts - 1], flavorlist.size() - 1, 0);	//选出合适的k

		if (k != -1) {
			flavorlist[k].num--;
			sumflavor--;
			serverlist[Server::counts - 1].cpu -= flavorlist[k].cpu;
			serverlist[Server::counts - 1].memory -= flavorlist[k].memory;
			serverlist[Server::counts - 1].flavornum[k]++;
		}
		else {
			serverlist.push_back(Server());
		}

	}

	return Server::counts;
}

说明：
首先求出虚拟机总数sumflavor，然后进入while循环，每次选出一个虚拟机分配在物理机上。
调用selectFlavor函数，根据返回值k作出选择。若k!=-1，则挑选对应的虚拟机进行分配；
若k=-1，表明这台物理机已经溢出了，就新建下一台物理机继续放置。

3.类设计
class Flavor {
public:
	int cpu;
	int memory;
	string name;
	int num;
public:
	//	Flavor() {}
	Flavor(int cpu, int memory, string name, int num = 0) : cpu(cpu), memory(memory), name(name), num(num) {}
	Flavor& operator=(const Flavor& flavor) {
		cpu = flavor.cpu;
		memory = flavor.memory;
		name = flavor.name;
		num = flavor.num;
		return *this;
	}
};

class Server {
public:
	int cpu;
	int memory;
	vector<int> flavornum;					//Server类内拥有的各虚拟机个数
	int id;
	Server() :cpu(Cpu), memory(Memory), flavornum(Prenum, 0), id(counts) {
		counts++;
	}
	~Server() {
		//	counts--;
	}
	static int counts;
};

int Server::counts = 0;

说明：
虚拟机类存放虚拟机的内存、cpu规格、名字、数量，以及构造函数和赋值操作符。
物理机类存放物理机的剩余内存、剩余cpu、编号，以及各种规格虚拟机的数量，内部还存放静态变量counts，每创建一台物理机counts自加，表明物理机总数。
