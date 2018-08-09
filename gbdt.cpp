#include "predict.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int isCpu = -1;				//是否优化cpu

static int Cpu;
static int Memory;
static int Prenum;			//预测虚拟机种类数目



class Flavor {
public:
	int cpu;
	int memory;
	string name;
	int num;
public:
	//	Flavor() {}
	Flavor(int cpu, int memory, string name, int num = 0) : cpu(cpu), memory(memory), name(name), num(num) {}
	Flavor& operator=(Flavor& flavor) {
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


void initFlavorAllList(vector<vector<int> >& flavoralllist, int flavornums, int days) {
	for (int i = 0; i < flavornums; i++) {
		flavoralllist.push_back(vector<int>(days, 0));
	}
}

void initFlavor(vector<Flavor>& flavorlist) {
	flavorlist.push_back(Flavor(1, 1024, "flavor1"));
	flavorlist.push_back(Flavor(1, 2048, "flavor2"));
	flavorlist.push_back(Flavor(1, 4096, "flavor3"));
	flavorlist.push_back(Flavor(2, 2048, "flavor4"));
	flavorlist.push_back(Flavor(2, 4096, "flavor5"));
	flavorlist.push_back(Flavor(2, 8192, "flavor6"));
	flavorlist.push_back(Flavor(4, 4096, "flavor7"));
	flavorlist.push_back(Flavor(4, 8192, "flavor8"));
	flavorlist.push_back(Flavor(4, 16384, "flavor9"));
	flavorlist.push_back(Flavor(8, 8192, "flavor10"));
	flavorlist.push_back(Flavor(8, 16384, "flavor11"));
	flavorlist.push_back(Flavor(8, 32768, "flavor12"));
	flavorlist.push_back(Flavor(16, 16384, "flavor13"));
	flavorlist.push_back(Flavor(16, 32768, "flavor14"));
	flavorlist.push_back(Flavor(16, 65536, "flavor15"));
}


//以下三个函数用来预测未来各虚拟机台数
class treenode {
public:
	treenode() : col(-1), value(0), left(nullptr), right(nullptr) {}
	treenode(int col, float value, vector<float> result) :
		col(col), value(value), left(nullptr), right(nullptr), result(result) { }


public:
	int col;		//以col作为特征划分
	float value;		//以小于等于value作为划分依据，满足则为左子树，否则为右子树
	treenode* left;
	treenode* right;
	vector<float> result; //只对叶节点存放输出值
};


pair<vector<vector<float> >, vector<vector<float> >> divideTwo(vector<vector<float> >& inputvector, int col, float value) {	//以col和value将输入分为两部分
	vector<vector<float> > leftvector;
	vector<vector<float> > rightvector;
	for (unsigned int i = 0; i < inputvector.size(); i++) {
		if (inputvector[i][col] <= value)
			leftvector.push_back(inputvector[i]);
		else
			rightvector.push_back(inputvector[i]);
	}
	pair<vector<vector<float> >, vector<vector<float> >> a(leftvector, rightvector);
	return a;
}

float squareSum(vector<vector<float> >& leftvector) {
	if (leftvector.size() == 0)
		return 0;
	float sum = 0;
	float sum2 = 0;
	for (unsigned int i = 0; i < leftvector.size(); i++) {
		sum += leftvector[i][leftvector[0].size() - 1];
		sum2 += (leftvector[i][leftvector[0].size() - 1] * leftvector[i][leftvector[0].size() - 1]);
	}
	//	cout << (sum2 - sum*sum / leftvector.size()) << endl;
	return (sum2 - sum*sum / leftvector.size());
}


vector<float> divideValue(vector<float> values) {		//不传引用
	if (values.size() <= 1)
		return values;
	vector<float> dividevalues;
	sort(values.begin(), values.end());
	float delta = (values[values.size() - 1] - values[0]) / (values.size() - 1);
	for (unsigned int i = 0; i < values.size(); i++) {
		dividevalues.push_back(values[0] + delta*i);
	}
	return dividevalues;
}


vector<float> getColValues(vector<vector<float> >& inputvector, unsigned int col) {
	vector<float> colvalues;
	for (unsigned int i = 0; i < inputvector.size(); i++) {
		colvalues.push_back(inputvector[i][col]);
	}
	return colvalues;
}




treenode* buildTree(vector<vector<float> >& inputvector) {							//可能会超时
	treenode *node = new treenode();
	float minsquaresum = squareSum(inputvector);
	if (inputvector.size() <= 1 || minsquaresum == 0.0) {								//叶结点,只有一个向量，或者输出相同
		for (unsigned int i = 0; i < inputvector.size(); i++)
			(node->result).push_back(inputvector[i][inputvector[0].size() - 1]);
		return node;
	}

	pair<vector<vector<float> >, vector<vector<float> >> a;
	vector<vector<float> > leftvector;
	vector<vector<float> > rightvector;

	vector<vector<float> > finalleftvector;
	vector<vector<float> > finalrightvector;


	float squaresum = 0.0;

	unsigned int thecol = -1;
	float value = 0.0;
	bool flag = false;				//正常情况下flag为true

	vector<float> colvalues;
	vector<float> sortvalues;

	for (unsigned col = 0; col < inputvector[0].size() - 1; col++) {
		colvalues = getColValues(inputvector, col);		//即y值
		sortvalues = divideValue(colvalues);
		for (unsigned j = 0; j < inputvector.size(); j++) {
			a = divideTwo(inputvector, col, sortvalues[j]);		//切分
			leftvector = a.first;
			rightvector = a.second;
			squaresum = squareSum(leftvector) + squareSum(rightvector);	//存在无法分开的情形
			if (squaresum < minsquaresum) {
				flag = true;
				minsquaresum = squaresum;
				thecol = col;
				value = sortvalues[j];
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




vector<vector<float> > getInputValues(vector<vector<float> >& inputvector) {
	vector<vector<float> > inputs;
	vector<float> row;
	for (unsigned int i = 0; i < inputvector.size(); i++) {
		for (unsigned int j = 0; j < inputvector[0].size() - 1; j++)
			row.push_back(inputvector[i][j]);
		inputs.push_back(row);
		row.clear();
	}
	return inputs;
}



vector<vector<float> > divideserial(vector<float>& flavor, int num) {//输入向量为num+1维的，其中最后一维为输出
	vector<float> input(num + 1, 0);
	vector<vector<float> > inputvector;
	for (unsigned int i = 0; i < flavor.size() - num; i++) {
		for (int j = 0; j < num + 1; j++) {
			input[j] = flavor[i + j];
		}
		inputvector.push_back(input);
	}
	return inputvector;
}


void drawTree(treenode& node, string space = " ") {
	if (node.col != -1) {			//分支节点
		cout << "(" << node.col << "," << node.value << ")" << endl;
		cout << space;
		cout << "left:";
		drawTree(*node.left, space + ' ');
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


float preOutput(treenode* node, vector<float>& input) {			//给定决策树，输入变量，求输出
	int col = node->col;
	float value = node->value;
	if (col == -1) {
		if (node->result.size() == 0) {
			cout << "node wrong!";
			return -1;
		}
		sort(node->result.begin(), node->result.end());
		return node->result[(node->result.size()) / 2];
		//	for (unsigned int i = 0; i < node->result.size(); i++) {
		//		sum += (node->result[i]);
		//	}
		//	return static_cast<int>(sum / (node->result.size()) + 0.5);
	}
	if (input[col] <= value)
		return preOutput(node->left, input);
	else
		return preOutput(node->right, input);
}


vector<float> continuesOutput(treenode* node, vector<float>& input, int days) {		//预测几天输出
	vector<float> predays;
	float result;
	for (int i = 0; i < days; i++) {
		result = preOutput(node, input);
		predays.push_back(result);
		for (unsigned int j = 0; j < input.size() - 1; j++)			//对输入左移
			input[j] = input[j + 1];
		input[input.size() - 1] = result;
	}
	return predays;
}


vector<float> getInput(vector<float>& input, int num) {	//num为输入向量维数
	vector<float> getinput;
	for (int i = 0; i < num; i++) {
		getinput.push_back(input[input.size() - num + i]);
	}
	return getinput;
}


float squareDiff(vector<float>& result) {		//方差
	if (result.size() == 0)
		return 0;
	float sum = 0;
	float sum2 = 0;
	for (unsigned int i = 0; i < result.size(); i++) {
		sum += result[i];
		sum2 += (result[i] * result[i]);
	}
	//	cout << (sum2 - sum*sum / leftvector.size()) << endl;
	return (sum2 - sum*sum / result.size()) / result.size();
}


void prune(treenode* node, float mingain = 0.9) {
	if (node->col == -1)
		return;
	if (node->left->col != -1)			//分支结点
		prune(node->left, mingain);
	if (node->right->col != -1)			//分支结点
		prune(node->right, mingain);

	if ((node->left->col == -1) && (node->right->col == -1)) {
		vector<float> mergeresult;
		float leftsquarediff = squareDiff(node->left->result);
		float rightsquarediff = squareDiff(node->right->result);
		for (unsigned int i = 0; i < node->left->result.size(); i++) {
			mergeresult.push_back(node->left->result[i]);
		}
		for (unsigned int i = 0; i < node->right->result.size(); i++) {
			mergeresult.push_back(node->right->result[i]);
		}
		float mergesquarediff = squareDiff(mergeresult);
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


float autocorr(vector<float>& serials, unsigned int m) {
	if (serials.size() - m <= 0)
		return -1;
	float sum = 0.0;
	for (unsigned int i = m; i < serials.size(); i++) {
		sum += serials[i] * serials[i - m];
	}
	return sum / (serials.size() - m);
}


float gbdtPreOutput(vector<treenode*> nodevector, vector<float>& input) {
	float sum = 0.0;
	for (unsigned int i = 0; i < nodevector.size(); i++) {
		//		cout << "gbdt决策树" << i << ":" << preOutput(nodevector[i], input) << endl;
		sum += preOutput(nodevector[i], input);

	}
	return sum;
}

vector<float> gbdtContinuesOutput(vector<treenode*> nodevector, vector<float>& input, int days) {		//预测几天输出
	vector<float> predays;
	float result;
	for (int i = 0; i < days; i++) {
		result = gbdtPreOutput(nodevector, input);
		predays.push_back(result);
		for (unsigned int j = 0; j < input.size() - 1; j++)			//对输入左移
			input[j] = input[j + 1];
		input[input.size() - 1] = result;
	}
	return predays;
}


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
			//		temp = preOutput(nodevector[k], inputs[t]);
//			cout << temp << ' ';
			inputvector[t][inputvector[0].size() - 1] = colvalues[t] - temp;	//计算残差
		}
//		cout << endl;

		vector<float> tempcolvalues = getColValues(inputvector, inputvector[0].size() - 1);
		float sumdif = squareDiff(tempcolvalues);
//		cout << "方差:" << sumdif << endl;

		node = buildTree(inputvector);
		prune(node, sumdif*rate);
		nodevector.push_back(node);
	}

	return nodevector;

}

void exceptionDetect(vector<float>& input, float rate = 3, float dif = 2) {	//rate为倍率，即高于平均值或低于平均值多少倍，则判为异常点
	if (input.size() <= 1)
		return;
	float sum = 0;
	float avg = 0.0;
	for (unsigned int i = 0; i < input.size(); i++) {
		sum += input[i];
	}
	for (unsigned int j = 0; j < input.size(); j++) {
		avg = (float)((sum - input[j]) / (input.size() - 1));
		if (avg == 0)
			return;
		if (input[j] / avg > rate && input[j] - avg > dif) {
			//			cout << input[j] << ":" << avg << endl;
			input[j] = avg;
		}
	}
}







//Dates类，方便填充日期
class Dates {
public:
	Dates() {}
	Dates(int year, int month, int day) : year(year), month(month), day(day) {}
	Dates operator++() {
		if (day < dayOfMonth()) {
			day++;
		}
		else if (day == dayOfMonth()) {
			day = 1;
			if (month < 12) {
				month++;
			}
			else if (month == 12) {
				month = 1;
				year++;
			}
		}
		return *this;
	}

	bool operator<(Dates & rhs) {
		if (year < rhs.year)
			return true;
		else if (year == rhs.year) {
			if (month < rhs.month)
				return true;
			else if (month == rhs.month) {
				if (day < rhs.day)
					return true;
			}
		}
		return false;
	}

	bool isRun() {
		if ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))
			return 1;
		else
			return 0;
	}


	int dayOfMonth() {
		switch (month) {
		case 1:	return 31;
		case 2:	return isRun() ? 29 : 28;
		case 3:	return 31;
		case 4: return 30;
		case 5: return 31;
		case 6: return 30;
		case 7: return 31;
		case 8: return 31;
		case 9: return 30;
		case 10: return 31;
		case 11: return 30;
		case 12: return 31;
		default: return -1;
		}
	}

public:
	int year;
	int month;
	int day;

};



Dates stringToDates(string& dates) {
	Dates temp;
	string year, month, day;
	int n1 = dates.find_first_of('-', 0);
	int n2 = dates.find_first_of('-', n1 + 1);
	year = dates.substr(0, n1);
	month = dates.substr(n1 + 1, n2 - n1 - 1);
	day = dates.substr(n2 + 1, 2);
	temp.year = atoi(year.c_str());
	temp.month = atoi(month.c_str());
	temp.day = atoi(day.c_str());
	return temp;
}

string datesToString(Dates& dates) {
	string year(to_string(dates.year));
	string month(to_string(dates.month));
	string day(to_string(dates.day));
	if (month.size() == 1)
		month = '0' + month;
	if (day.size() == 1)
		day = '0' + day;
	return year + '-' + month + '-' + day;

}


vector<string> getDate(string& start, string& finish) {
	vector<string> dates;
	Dates startdate;
	Dates finishdate;
	startdate = stringToDates(start);
	finishdate = stringToDates(finish);
	while (startdate < finishdate) {
		dates.push_back(datesToString(startdate));
		++startdate;
	}
	dates.push_back(finish);
	return dates;
}


float sum(vector<float>& abc) {
	float s = 0.0;
	for (unsigned int i = 0; i < abc.size(); i++)
		s += abc[i];
	return s;
}


//以下三个为预测函数
int simCalculate(Flavor& flavor, Server& server) {
	int tmp = static_cast<int>((Memory - server.memory + flavor.memory) / (Cpu - server.cpu + flavor.cpu));
	return abs(tmp - (static_cast<int>(Memory / Cpu)));
}

int selectFlavor(vector<Flavor>& flavorlist, Server& server, int last, int first) {    //选出合适的k
	int minmum = 5000;
	int maxmum = 1000;
	int k = last;

	if (last < first)
		return -1;

	for (int i = last; i >= first; i--) {		//寻找k
		if (flavorlist[i].num > 0) {
			if (server.cpu == Cpu || isCpu *(Memory - server.memory) / (Cpu - server.cpu) < isCpu*(Memory / Cpu)) {		//最不平衡
				int tmpmax = static_cast<int>(flavorlist[i].memory / flavorlist[i].cpu);
				if (tmpmax > maxmum) {
					k = i;
					maxmum = tmpmax;
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

//你要完成的功能总入口
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
	vector<string> strings;			//每行数据
	vector<string> flavors;			//分离strings得到每行对应的flavor
	vector<string> dates;			//分离strings得到每行对应的date,包含日期,去除时间

	vector<string> infostrings;

	Dates tempdate;
	vector<string> alldates;				//所有连续的天数
	vector<vector<int> > flavoralllist;		//每种虚拟机每天的数目
	vector<Flavor> flavorlist;				//每种虚拟机的规格
	vector<Flavor> flavortopre;
	vector<string> flavorname;

	int num = 0;				//需预测天数
//	int servernum = 0;

	vector<Server> serverlist;

	//解析训练数据
	for (int i = 0; i < data_num; i++) {
		strings.push_back(string(data[i]));						//包含\n
		int n1 = strings[i].find_first_of('\t', 0);
		int n2 = strings[i].find_first_of('\t', n1 + 1);
		int n3 = strings[i].find_first_of(' ', 0);
		flavors.push_back(strings[i].substr(n1 + 1, n2 - n1 - 1));
		dates.push_back(strings[i].substr(n2 + 1, n3 - n2 - 1));
	}


	//解析信息头
	string serverinfo(info[0]);					//提取服务器信息
	int m1 = serverinfo.find_first_of(' ', 0);
	string CPU = serverinfo.substr(0, m1);
	int m2 = serverinfo.find_first_of(' ', m1+1);
	string MEMORY = serverinfo.substr(m1 + 1, m2 - m1 - 1);
	Cpu = atoi(CPU.c_str());
	Memory = atoi(MEMORY.c_str()) * 1024;
	string PRENUM(info[2]);
	int m3 = PRENUM.find_first_of('\n', 0);
	Prenum = atoi(PRENUM.substr(0, m3).c_str());

	for (int i = 3; i < Prenum + 3; i++) {
		string tempinfo(info[i]);
		int m3 = tempinfo.find_first_of(' ', 0);
		int m4 = tempinfo.find_first_of(' ', m3 + 1);
		int m5 = tempinfo.find_first_of('\n', m4 + 1);
		string theflavor(tempinfo.substr(0, m3));
		string thecpu(tempinfo.substr(m3 + 1, m4 - m3 - 1));
		string thememory(tempinfo.substr(m4 + 1, m5 - m4 - 1));
		flavortopre.push_back(Flavor(atoi(thecpu.c_str()), atoi(thememory.c_str()), theflavor));
	}

	string ISCPU(info[Prenum + 4]);
	int m6 = ISCPU.find_first_of('\n', 0);
	string ISCPUSTR = ISCPU.substr(0, m6);
	if (ISCPUSTR == string("CPU")) 
		isCpu = 1;
	else
		isCpu = -1;

	//获取时间间隔
	string DATESTART(info[Prenum + 6]);
	string DATEFINISH(info[Prenum + 7]);
	Dates tempstart, tempfinish;
	string year, month, day;

	int n1 = DATESTART.find_first_of('-', 0);
	int n2 = DATESTART.find_first_of('-', n1 + 1);
	year = DATESTART.substr(0, n1);
	month = DATESTART.substr(n1 + 1, n2 - n1 - 1);
	day = DATESTART.substr(n2 + 1, 2);
	tempstart.year = atoi(year.c_str());
	tempstart.month = atoi(month.c_str());
	tempstart.day = atoi(day.c_str());

	int n3 = DATEFINISH.find_first_of('-', 0);
	int n4 = DATEFINISH.find_first_of('-', n3 + 1);
	year = DATEFINISH.substr(0, n3);
	month = DATEFINISH.substr(n3 + 1, n4 - n3 - 1);
	day = DATEFINISH.substr(n4 + 1, 2);
	tempfinish.year = atoi(year.c_str());
	tempfinish.month = atoi(month.c_str());
	tempfinish.day = atoi(day.c_str());

	while (tempstart < tempfinish) {
		num++;										//得到需预测天数
		++tempstart;
	}

	vector<int> predicateday(num, 0);


	alldates = getDate(dates[0], dates[dates.size() - 1]);

	initFlavorAllList(flavoralllist, 15, alldates.size());			//15虚拟机种类待修改，初始化每种虚拟机每日数量0
	initFlavor(flavorlist);											//初始化每种虚拟机规格

	for (unsigned int i = 0; i < flavorlist.size(); i++)
		flavorname.push_back(flavorlist[i].name);					//初始化flavorname

	for (unsigned int line = 0; line < dates.size(); line++) {				//求出flavoralllist
		int indexflavor = 0;
		int indexday = 0;
		vector<string>::iterator itflavor = find(flavorname.begin(), flavorname.end(), flavors[line]);
		if (itflavor != flavorname.end()) {
			indexflavor = distance(flavorname.begin(), itflavor);	//找到对应flavor的索引
			vector<string>::iterator itday = find(alldates.begin(), alldates.end(), dates[line]);
			indexday = distance(alldates.begin(), itday);			//找到对应day的索引
			flavoralllist[indexflavor][indexday]++;
		}
	}


	vector<vector<float> > inputvector;
//	treenode *node;
	vector<float> inputtest;
	int dimension = 7;
	vector<float> resultdays;

//int转float
	vector<vector<float> > floatalllist;
	vector<float> floatlist;
	for (unsigned int u = 0; u < flavoralllist.size(); u++) {
		for (unsigned int p = 0; p < flavoralllist[0].size(); p++) {
			floatlist.push_back((float)(flavoralllist[u][p]));
		}
		floatalllist.push_back(floatlist);
		floatlist.clear();
	}




	for (unsigned int i = 0; i < flavorlist.size(); i++) {									//对flavor对象的num赋值，即预测过程

		float max = 0.0;
		float corr;
		exceptionDetect(floatalllist[i], 6, 1);
		for (unsigned int m = 3; m < 15; m++) {
			corr = autocorr(floatalllist[i], m);
			if (corr > max) {
				max = corr;
				dimension = m;
			}
		}
		cout << "维数:" << dimension << endl;

		inputvector = divideserial(floatalllist[i], dimension);
		
		/*node = buildTree(inputvector);
		prune(node);
		drawTree(*node);*/

		vector<treenode*> nodevector;
		nodevector = gbdt(inputvector, 3);

		inputtest = getInput(floatalllist[i], dimension);


		resultdays = gbdtContinuesOutput(nodevector, inputtest, num);

		int tempnum = (int)(sum(resultdays)+0.5);
		flavorlist[i].num = tempnum;
		cout << flavorlist[i].name << ' ' << flavorlist[i].num << endl;
		for (unsigned int j = 0; j < flavortopre.size(); j++) {
			if (flavortopre[j].name == flavorlist[i].name)
				flavortopre[j] = flavorlist[i];
		}
	}


	//写入预测出的虚拟机数量信息
	int sums = 0;
	for (unsigned int i = 0; i < flavortopre.size(); i++) {
		sums += flavortopre[i].num;
	}
	string flavorinfo(to_string(sums));
	flavorinfo = flavorinfo + '\n';
	for (unsigned int i = 0; i < flavortopre.size(); i++) {
		flavorinfo = flavorinfo + flavortopre[i].name + ' ' + to_string(flavortopre[i].num) + '\n';
	}
	flavorinfo = flavorinfo + '\n';


	/*for (unsigned int g = 0; g < flavortopre.size(); g++) {
		cout << flavortopre[g].name << ' ' << flavortopre[g].cpu << ' '
			<< flavortopre[g].memory << ' ' << flavortopre[g].num << endl;
	}*/

	(void)allocate(flavortopre, serverlist);					//分配过程


																	//写入服务器的分配信息
	int serversums = serverlist.size();
	string serverinfos(to_string(serversums));
	serverinfos = serverinfos + '\n';
	for (unsigned int i = 0; i < serverlist.size(); i++) {
		serverinfos = serverinfos + to_string(serverlist[i].id + 1) + ' ';
		for (unsigned int j = 0; j < serverlist[i].flavornum.size(); j++) {
			if (serverlist[i].flavornum[j] != 0) {
				serverinfos = serverinfos + flavortopre[j].name + ' ' + to_string(serverlist[i].flavornum[j]) + ' ';
			}
		}
		if (i != serverlist.size() - 1)
			serverinfos = serverinfos + '\n';
	}

	string finalinfo;
	finalinfo = flavorinfo + serverinfos;



	//for (unsigned int i = 0; i < serverlist.size(); i++) {					//打印分配信息
	//	int cpupro = 0;
	//	int memorypro = 0;
	//	cout << "Serverid" << serverlist[i].id << ": ";
	//	for (int j = 0; j < Prenum; j++) {
	//		cout << serverlist[i].flavornum[j] << ' ';
	//		cpupro += serverlist[i].flavornum[j] * flavorlist[j].cpu;
	//		memorypro += serverlist[i].flavornum[j] * flavorlist[j].memory;
	//	}
	//	cout << endl;
	//	cout << "cpu: " << static_cast<float>(cpupro / 56.0) * 100 << "% memory: " << static_cast<float>(memorypro / 131072.0) * 100 << "% " << endl;
	//}


	// 需要输出的内容
	char * result_file = (char *)finalinfo.c_str();

	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	write_result(result_file, filename);
}
