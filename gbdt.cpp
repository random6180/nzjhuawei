#include "predict.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;

int isCpu = -1;				//�Ƿ��Ż�cpu

static int Cpu;
static int Memory;
static int Prenum;			//Ԥ�������������Ŀ



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
	vector<int> flavornum;					//Server����ӵ�еĸ����������
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


//����������������Ԥ��δ���������̨��
class treenode {
public:
	treenode() : col(-1), value(0), left(nullptr), right(nullptr) {}
	treenode(int col, float value, vector<float> result) :
		col(col), value(value), left(nullptr), right(nullptr), result(result) { }


public:
	int col;		//��col��Ϊ��������
	float value;		//��С�ڵ���value��Ϊ�������ݣ�������Ϊ������������Ϊ������
	treenode* left;
	treenode* right;
	vector<float> result; //ֻ��Ҷ�ڵ������ֵ
};


pair<vector<vector<float> >, vector<vector<float> >> divideTwo(vector<vector<float> >& inputvector, int col, float value) {	//��col��value�������Ϊ������
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


vector<float> divideValue(vector<float> values) {		//��������
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




treenode* buildTree(vector<vector<float> >& inputvector) {							//���ܻᳬʱ
	treenode *node = new treenode();
	float minsquaresum = squareSum(inputvector);
	if (inputvector.size() <= 1 || minsquaresum == 0.0) {								//Ҷ���,ֻ��һ�����������������ͬ
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
	bool flag = false;				//���������flagΪtrue

	vector<float> colvalues;
	vector<float> sortvalues;

	for (unsigned col = 0; col < inputvector[0].size() - 1; col++) {
		colvalues = getColValues(inputvector, col);		//��yֵ
		sortvalues = divideValue(colvalues);
		for (unsigned j = 0; j < inputvector.size(); j++) {
			a = divideTwo(inputvector, col, sortvalues[j]);		//�з�
			leftvector = a.first;
			rightvector = a.second;
			squaresum = squareSum(leftvector) + squareSum(rightvector);	//�����޷��ֿ�������
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

	if (flag == false) {								//Ҷ���,ֻ��һ�����������������ͬ
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



vector<vector<float> > divideserial(vector<float>& flavor, int num) {//��������Ϊnum+1ά�ģ��������һάΪ���
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
	if (node.col != -1) {			//��֧�ڵ�
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


float preOutput(treenode* node, vector<float>& input) {			//��������������������������
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


vector<float> continuesOutput(treenode* node, vector<float>& input, int days) {		//Ԥ�⼸�����
	vector<float> predays;
	float result;
	for (int i = 0; i < days; i++) {
		result = preOutput(node, input);
		predays.push_back(result);
		for (unsigned int j = 0; j < input.size() - 1; j++)			//����������
			input[j] = input[j + 1];
		input[input.size() - 1] = result;
	}
	return predays;
}


vector<float> getInput(vector<float>& input, int num) {	//numΪ��������ά��
	vector<float> getinput;
	for (int i = 0; i < num; i++) {
		getinput.push_back(input[input.size() - num + i]);
	}
	return getinput;
}


float squareDiff(vector<float>& result) {		//����
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
	if (node->left->col != -1)			//��֧���
		prune(node->left, mingain);
	if (node->right->col != -1)			//��֧���
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
		if (mergesquarediff - leftsquarediff - rightsquarediff < mingain) {		//�����С����
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
		//		cout << "gbdt������" << i << ":" << preOutput(nodevector[i], input) << endl;
		sum += preOutput(nodevector[i], input);

	}
	return sum;
}

vector<float> gbdtContinuesOutput(vector<treenode*> nodevector, vector<float>& input, int days) {		//Ԥ�⼸�����
	vector<float> predays;
	float result;
	for (int i = 0; i < days; i++) {
		result = gbdtPreOutput(nodevector, input);
		predays.push_back(result);
		for (unsigned int j = 0; j < input.size() - 1; j++)			//����������
			input[j] = input[j + 1];
		input[input.size() - 1] = result;
	}
	return predays;
}


vector<treenode*> gbdt(vector<vector<float> >& inputvector, int iter, float rate=1) {
	vector<treenode*> nodevector;
	treenode* node = new treenode;		//nodeΪ����
	vector<float> results;
	vector<float> colvalues;
	vector<float> sortvalues;
	float sum2 = 0.0;
	float min = 10000000;
	float f0;

	colvalues = getColValues(inputvector, inputvector[0].size() - 1);		//��yֵ
	sortvalues = divideValue(colvalues);					//���ȷָ�����	
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
//		cout << "��������:" << k << endl;
		for (unsigned int t = 0; t < inputvector.size(); t++) {
			temp = gbdtPreOutput(nodevector, inputs[t]);
			//		temp = preOutput(nodevector[k], inputs[t]);
//			cout << temp << ' ';
			inputvector[t][inputvector[0].size() - 1] = colvalues[t] - temp;	//����в�
		}
//		cout << endl;

		vector<float> tempcolvalues = getColValues(inputvector, inputvector[0].size() - 1);
		float sumdif = squareDiff(tempcolvalues);
//		cout << "����:" << sumdif << endl;

		node = buildTree(inputvector);
		prune(node, sumdif*rate);
		nodevector.push_back(node);
	}

	return nodevector;

}

void exceptionDetect(vector<float>& input, float rate = 3, float dif = 2) {	//rateΪ���ʣ�������ƽ��ֵ�����ƽ��ֵ���ٱ�������Ϊ�쳣��
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







//Dates�࣬�����������
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


//��������ΪԤ�⺯��
int simCalculate(Flavor& flavor, Server& server) {
	int tmp = static_cast<int>((Memory - server.memory + flavor.memory) / (Cpu - server.cpu + flavor.cpu));
	return abs(tmp - (static_cast<int>(Memory / Cpu)));
}

int selectFlavor(vector<Flavor>& flavorlist, Server& server, int last, int first) {    //ѡ�����ʵ�k
	int minmum = 5000;
	int maxmum = 1000;
	int k = last;

	if (last < first)
		return -1;

	for (int i = last; i >= first; i--) {		//Ѱ��k
		if (flavorlist[i].num > 0) {
			if (server.cpu == Cpu || isCpu *(Memory - server.memory) / (Cpu - server.cpu) < isCpu*(Memory / Cpu)) {		//�ƽ��
				int tmpmax = static_cast<int>(flavorlist[i].memory / flavorlist[i].cpu);
				if (tmpmax > maxmum) {
					k = i;
					maxmum = tmpmax;
				}
			}
			else {						//��ƽ��
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
		return selectFlavor(flavorlist, server, k - 1, first);		//�ݹ�Ѱ��
}


int allocate(vector<Flavor>& flavorlist, vector<Server>& serverlist) {
	int sumflavor = 0;										//���������
	int k = 0;												//�����������

	for (unsigned int i = 0; i < flavorlist.size(); i++) {
		sumflavor += flavorlist[i].num;
	}

	serverlist.push_back(Server());
	while (sumflavor > 0) {
		k = selectFlavor(flavorlist, serverlist[Server::counts - 1], flavorlist.size() - 1, 0);	//ѡ�����ʵ�k

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

//��Ҫ��ɵĹ��������
void predict_server(char * info[MAX_INFO_NUM], char * data[MAX_DATA_NUM], int data_num, char * filename)
{
	vector<string> strings;			//ÿ������
	vector<string> flavors;			//����strings�õ�ÿ�ж�Ӧ��flavor
	vector<string> dates;			//����strings�õ�ÿ�ж�Ӧ��date,��������,ȥ��ʱ��

	vector<string> infostrings;

	Dates tempdate;
	vector<string> alldates;				//��������������
	vector<vector<int> > flavoralllist;		//ÿ�������ÿ�����Ŀ
	vector<Flavor> flavorlist;				//ÿ��������Ĺ��
	vector<Flavor> flavortopre;
	vector<string> flavorname;

	int num = 0;				//��Ԥ������
//	int servernum = 0;

	vector<Server> serverlist;

	//����ѵ������
	for (int i = 0; i < data_num; i++) {
		strings.push_back(string(data[i]));						//����\n
		int n1 = strings[i].find_first_of('\t', 0);
		int n2 = strings[i].find_first_of('\t', n1 + 1);
		int n3 = strings[i].find_first_of(' ', 0);
		flavors.push_back(strings[i].substr(n1 + 1, n2 - n1 - 1));
		dates.push_back(strings[i].substr(n2 + 1, n3 - n2 - 1));
	}


	//������Ϣͷ
	string serverinfo(info[0]);					//��ȡ��������Ϣ
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

	//��ȡʱ����
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
		num++;										//�õ���Ԥ������
		++tempstart;
	}

	vector<int> predicateday(num, 0);


	alldates = getDate(dates[0], dates[dates.size() - 1]);

	initFlavorAllList(flavoralllist, 15, alldates.size());			//15�����������޸ģ���ʼ��ÿ�������ÿ������0
	initFlavor(flavorlist);											//��ʼ��ÿ����������

	for (unsigned int i = 0; i < flavorlist.size(); i++)
		flavorname.push_back(flavorlist[i].name);					//��ʼ��flavorname

	for (unsigned int line = 0; line < dates.size(); line++) {				//���flavoralllist
		int indexflavor = 0;
		int indexday = 0;
		vector<string>::iterator itflavor = find(flavorname.begin(), flavorname.end(), flavors[line]);
		if (itflavor != flavorname.end()) {
			indexflavor = distance(flavorname.begin(), itflavor);	//�ҵ���Ӧflavor������
			vector<string>::iterator itday = find(alldates.begin(), alldates.end(), dates[line]);
			indexday = distance(alldates.begin(), itday);			//�ҵ���Ӧday������
			flavoralllist[indexflavor][indexday]++;
		}
	}


	vector<vector<float> > inputvector;
//	treenode *node;
	vector<float> inputtest;
	int dimension = 7;
	vector<float> resultdays;

//intתfloat
	vector<vector<float> > floatalllist;
	vector<float> floatlist;
	for (unsigned int u = 0; u < flavoralllist.size(); u++) {
		for (unsigned int p = 0; p < flavoralllist[0].size(); p++) {
			floatlist.push_back((float)(flavoralllist[u][p]));
		}
		floatalllist.push_back(floatlist);
		floatlist.clear();
	}




	for (unsigned int i = 0; i < flavorlist.size(); i++) {									//��flavor�����num��ֵ����Ԥ�����

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
		cout << "ά��:" << dimension << endl;

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


	//д��Ԥ����������������Ϣ
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

	(void)allocate(flavortopre, serverlist);					//�������


																	//д��������ķ�����Ϣ
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



	//for (unsigned int i = 0; i < serverlist.size(); i++) {					//��ӡ������Ϣ
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


	// ��Ҫ���������
	char * result_file = (char *)finalinfo.c_str();

	// ֱ�ӵ�������ļ��ķ��������ָ���ļ���(ps��ע���ʽ����ȷ�ԣ�����н⣬��һ��ֻ��һ�����ݣ��ڶ���Ϊ�գ������п�ʼ���Ǿ�������ݣ�����֮����һ���ո�ָ���)
	write_result(result_file, filename);
}
