#include "predict.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

int isCpu = -1;				//是否优化cpu

static int Cpu;
static int Memory;
static int Prenum;			//预测虚拟机种类数目


class Matrix {
private:
	vector<vector<double> > bivector;
public:
	Matrix() { }

	Matrix(int m, int n) {
		vector<vector<double> > tmp(m, vector<double>(n, 0));
		bivector = tmp;
	}

	Matrix(const vector<vector<double> >& rhs) {
		vector<vector<double> > tmp(rhs.size(), vector<double>(rhs[0].size(), 0));
		for (unsigned int i = 0; i < rhs.size(); i++) {
			for (unsigned int j = 0; j < rhs[0].size(); j++)
				tmp[i][j] = rhs[i][j];
		}
		bivector = tmp;
	}
	Matrix(const Matrix& A) {
		bivector.clear();
		vector<double> tmp;
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getColumn(); j++) {
				tmp.push_back(A.bivector[i][j]);
			}
			bivector.push_back(tmp);
			tmp.clear();
		}
	}

	~Matrix() {
		bivector.clear();
	}

	const int getRow() const {
		return bivector.size();

	}
	const int getColumn() const {
		return bivector[0].size();
	}

	Matrix& operator = (const Matrix& rhs) {
		if (this == &rhs) return *this;
		bivector.clear();
		vector<double> tmp;
		for (int i = 0; i < rhs.getRow(); i++) {
			for (int j = 0; j < rhs.getColumn(); j++) {
				tmp.push_back(rhs.bivector[i][j]);
			}
			bivector.push_back(tmp);
			tmp.clear();
		}
		return *this;
	}

	bool Matrix::operator == (const Matrix& rhs) {
		if (this->getRow() != rhs.getRow() || this->getColumn() != rhs.getColumn())
			return false;
		else {
			for (int i = 0; i < rhs.getRow(); i++) {
				for (int j = 0; j < rhs.getColumn(); j++) {
					if (abs(this->bivector[i][j] - rhs.bivector[i][j]) > 1e-10)
						return false;
				}
			}
		}
		return true;
	}

	double & operator()(int i, int j) {
		return this->bivector[i - 1][j - 1];
	}

	/*double & operator()(int i, int j) const {
		return this->bivector[i - 1][j - 1];
	}*/

	Matrix Transpose() {
		Matrix aaa(this->getColumn(), this->getRow());
		for (int i = 0; i < this->getRow(); i++) {
			for (int j = 0; j < this->getColumn(); j++)
				aaa.bivector[j][i] = this->bivector[i][j];
		}
		return aaa;
	}

	Matrix Inverse() {
		Matrix tmp;
		Matrix A;
		A = this->Adjugate();
		tmp = A;
		tmp *= (1 / this->det());
		return tmp;
	}

	double det() {
		int row = this->getRow();
		double tmp = 1;
		Matrix L2, U2;
		this->LU(L2, U2);
		for (int i = 0; i < row; i++) {
			tmp = tmp*L2.bivector[i][i] * U2.bivector[i][i];
		}
		return tmp;
	}

	Matrix Adjugate() //Adjoint/Adjugate
	{
		Matrix tmp;
		tmp = this->Cofactor();
		tmp = tmp.Transpose();
		return tmp;
	}

	double Cofactors(int i, int j) {
		double tmp;
		tmp = this->Minor(i, j);
		tmp = pow(-1, (i + j))*tmp;
		return tmp;
	}

	Matrix Cofactor()//matrix of cofactors
	{
		Matrix tmp(this->getRow(), this->getColumn());
		for (int i = 1; i <= this->getRow(); i++)
		{
			for (int j = 1; j <= this->getColumn(); j++)
			{
				tmp(i, j) = this->Cofactors(i, j);
			}
		}
		return tmp;
	}

	double Minor(int i, int j) {
		double tmp;
		vector<double> onerow;
		vector<vector<double> > xxx;
		for (int m = 1; m <= this->getRow(); m++)
		{
			if (m == i) continue;
			for (int n = 1; n <= this->getColumn(); n++) {
				if (n == j) continue;
				onerow.push_back(this->bivector[m - 1][n - 1]);
			}
			xxx.push_back(onerow);
			onerow.clear();
		}
		Matrix A(xxx);
		tmp = A.det();
		return tmp;
	}

	bool LU(Matrix& L, Matrix& U) {
		vector<vector<double> > test(this->getRow(), vector<double>(this->getColumn(), 0));

		L.bivector = test;
		U.bivector = test;

		double sum = 0.0;

		for (long n = 0; n < this->getRow(); n++)
		{

			for (long j = n; j < U.getColumn(); j++)
			{
				sum = this->bivector[n][j];
				for (long k = 0; k < n; k++)
					sum -= (L.bivector[n][k])*(U.bivector[k][j]);
				U.bivector[n][j] = sum;
			}

			for (long i = 0; i < L.getRow(); i++)
			{
				sum = this->bivector[i][n];
				for (long k = 0; k < n; k++)
					sum -= (L.bivector[i][k])*(U.bivector[k][n]);
				if (U.bivector[n][n] == 0)
				{
					L.bivector.clear();
					U.bivector.clear();
					return false;
				}
				else
				{
					L.bivector[i][n] = sum*1.0 / (U.bivector[n][n]);
				}
			}
		}
		return true;
	}

	Matrix& operator += (const Matrix& A) {
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getColumn(); j++) {
				this->bivector[i][j] += A.bivector[i][j];
			}
		}
		return *this;
	}

	Matrix& operator -=(const Matrix& A) {
		for (int i = 0; i < A.getRow(); i++) {
			for (int j = 0; j < A.getColumn(); j++) {
				this->bivector[i][j] -= A.bivector[i][j];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator *=(const Matrix& A) {
		Matrix tmp(this->getRow(), A.getColumn());
		for (int i = 1; i <= tmp.getRow(); i++)
		{
			for (int j = 1; j <= tmp.getColumn(); j++)
			{
				tmp(i, j) = 0;
				for (int k = 1; k <= A.getRow(); k++)
					tmp(i, j) += (*this)(i, k) * A.bivector[k - 1][j - 1];
			}
		}
		*this = tmp;
		return *this;
	}

	Matrix& operator *=(double a)
	{
		for (int i = 0; i < this->getRow(); i++) {
			for (int j = 0; j < this->getColumn(); j++) {
				this->bivector[i][j] *= a;
			}
		}
		return *this;
	}

};


Matrix operator + (const Matrix& A, const Matrix& B)
{
	Matrix tmp = A;
	tmp += B;
	return tmp;
}

Matrix operator - (const Matrix& A, const Matrix& B)
{
	Matrix tmp = A;
	tmp -= B;
	return tmp;
}

Matrix operator * (const Matrix& A, const Matrix& B)
{
	Matrix tmp = A;
	tmp *= B;
	return tmp;
}


Matrix operator * (double a, const Matrix& A)
{
	Matrix tmp = A;
	tmp *= a;
	return tmp;
};


Matrix operator * (const Matrix& A, double a)
{
	Matrix tmp = A;
	tmp *= a;
	return tmp;
};


double diffSquareSum(vector<double>& op1, vector<double>& op2) {	//两个数组差的平方和，即距离的平方
	double sum = 0;
	for (unsigned int i = 0; i < op1.size(); i++)
		sum += (op1[i] - op2[i])*(op1[i] - op2[i]);
	return sum;
}

double squareDiff(vector<double>& result) {		//方差
	if (result.size() == 0)
		return 0;
	double sum = 0;
	double sum2 = 0;
	for (unsigned int i = 0; i < result.size(); i++) {
		sum += result[i];
		sum2 += (result[i] * result[i]);
	}
	//	cout << (sum2 - sum*sum / leftvector.size()) << endl;
	return (sum2 - sum*sum / result.size()) / result.size();
}


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


vector<double> lwlrTest(vector<vector<double> >& testArr, vector<vector<double> >& xArr,
	vector<vector<double> >& yArr, double k = 1.0) {
	vector<double> resultvector;
	double temp = 0;
	int m = testArr.size();
	for (int i = 0; i < m; i++) {
		temp = lwlr(testArr[i], xArr, yArr, k);
		resultvector.push_back(temp);
	}
	return resultvector;
}


double preOneDay(vector<double>& flavorlist, double day, double k = 1) {
	vector<double> testPoint;
	vector<vector<double> > xArr(flavorlist.size(), vector<double>(2, 0));				//输入天数从1开始计数
	vector<vector<double> > yArr(flavorlist.size(), vector<double>(1, 0));

	testPoint.push_back(1);
	testPoint.push_back(day);

	for (unsigned int i = 0; i < flavorlist.size(); i++) {
		xArr[i][0] = 1;
		xArr[i][1] = i + 1;
		yArr[i][0] = flavorlist[i];
	}

	double result;
	result = lwlr(testPoint, xArr, yArr, k);
	return result;
}


vector<double> preServeralDays(vector<double>& flavorlist, vector<double>& days, double k = 1) {
	vector<double> resultvector;
	double result;

	vector<vector<double> > xArr(flavorlist.size(), vector<double>(2, 0));				//输入天数从1开始计数
	vector<vector<double> > yArr(flavorlist.size(), vector<double>(1, 0));

	for (unsigned int i = 0; i < flavorlist.size(); i++) {
		xArr[i][0] = 1;
		xArr[i][1] = i + 1;
		yArr[i][0] = flavorlist[i];
	}


	vector<double> testPoint;
	for (unsigned int j = 0; j < days.size(); j++) {
		testPoint.push_back(1);
		testPoint.push_back(days[j]);
		result = lwlr(testPoint, xArr, yArr, k);
		resultvector.push_back(result);
		testPoint.clear();
	}
	return resultvector;
}


vector<double> sumflavor(vector<double>& flavorlist) {
	vector<double> sumflavorlist(flavorlist.size(), 0);
	sumflavorlist[0] = flavorlist[0];
	for (unsigned int i = 1; i < flavorlist.size(); i++)
		sumflavorlist[i] = sumflavorlist[i - 1] + flavorlist[i];
	return sumflavorlist;
}



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


int sum(vector<int>& abc) {
	int s = 0;
	for (unsigned int i = 0; i < abc.size(); i++)
		s += abc[i];
	return s;
}


//以下三个为预测函数
int simCalculate(Flavor& flavor, Server& server) {
	int tmp = (int)((Memory - server.memory + flavor.memory) / (Cpu - server.cpu + flavor.cpu));
	return abs(tmp - (int)(Memory / Cpu));
}

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

//一元非线性回归
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


void exceptionDetect(vector<double>& input) {	//rate为倍率，即高于平均值或低于平均值多少倍，则判为异常点
	if (input.size() <= 1)
		return;
	double sum = 0;
	//	double avg = 0.0;
	double squarediff = 0;
	vector<unsigned int> index;
	vector<double> days;
	vector<double> inputdup;
	vector<double> theta;
	inputdup = input;

	for (unsigned int i = 0; i < input.size(); i++) {
		sum += input[i];
	}

	//	avg = sum / input.size();
	//	squarediff = sqrt(squareDiff(input));		//标准差


	for (unsigned int j = 0; j < input.size(); j++) {
		double sumj = sum - input[j];
		double avgj = sumj / (input.size() - 1);
		vector<double> tempvector = input;
		tempvector.erase(tempvector.begin() + j);
		squarediff = sqrt(squareDiff(tempvector));
		if (abs(input[j] - avgj) > 5 * squarediff) {		//三倍标准差异常点检测
			index.push_back(j);						//异常点索引
		}
	}

	if (index.size() < 1) {
		cout << "No exception!" << endl;		//无异常点
		return;
	}

	for (unsigned int k = 0; k < input.size(); k++) {
		if (find(index.begin(), index.end(), k) != index.end()) {	//k值为异常点
			inputdup.erase(inputdup.begin() + k);
		}
		else {
			days.push_back(k + 1);
		}
	}

	double values;

	for (unsigned int m = 0; m < index.size(); m++) {
		theta = generatetheta(days, inputdup, (double)index[m] - 1, 1, 4);
		values = preOutput(theta, (double)index[m] + 1);
		//		cout << "day:" << index[m] + 1 << ' ' << "原值:" << input[index[m]] << ' ' << "修正值:" << values << endl;
		if (values < 0)
			values = 0;
		input[index[m]] = values;
	}

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
	vector<Server> serverlist1;
	vector<Server> serverlist2;

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
	int m2 = serverinfo.find_first_of(' ', m1 + 1);
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


	//int转double
	vector<vector<double> > floatalllist;
	vector<double> floatlist;
	for (unsigned int u = 0; u < flavoralllist.size(); u++) {
		for (unsigned int p = 0; p < flavoralllist[0].size(); p++) {
			floatlist.push_back((double)(flavoralllist[u][p]));
		}
		floatalllist.push_back(floatlist);
		floatlist.clear();
	}


	int days = flavoralllist[0].size();
	vector<double> dayvector;
	vector<double> sumflavorlist;
	//for (int ui = 0; ui < days; ui++) {
	//	dayvector.push_back((double)(ui + 1));			//生成从1开始的天数
	//}

	vector<double> theta;


	vector<double> resultdays;
	double result = 0;



	for (unsigned int i = 0; i < flavorlist.size(); i++) {	//对flavor对象的num赋值，即预测过程
		for (int ui = 0; ui < days; ui++) {
			dayvector.push_back((double)(ui + 1));			//生成从1开始的天数
		}
		//		cout << "flavor" << i + 1 << endl;
		exceptionDetect(floatalllist[i]);
		sumflavorlist = sumflavor(floatalllist[i]);
		double lastday = sumflavorlist[sumflavorlist.size() - 1];

		for (int daynum = 0; daynum < num; daynum++) {

			double xp = floatalllist[0].size() + daynum + 1;
			theta = generatetheta(dayvector, sumflavorlist, xp, 1, 3);
			result = preOutput(theta, xp);
			//			cout << "flavor" << i + 1 << ' ' << "result:" << result << endl;
			resultdays.push_back(result);

		}

		for (int daynum = 0; daynum < num; daynum++) {
			double xp = floatalllist[0].size() + daynum + 1;
			sumflavorlist.push_back(resultdays[daynum]);
			dayvector.push_back(xp);
		}

		resultdays.clear();
		for (int daynum = 0; daynum < num; daynum++) {

			double xp = floatalllist[0].size() + daynum + 1;
			theta = generatetheta(dayvector, sumflavorlist, xp - 4, 3, 20);
			result = preOutput(theta, xp);
			resultdays.push_back((int)(result + 0.5));

		}

		dayvector.clear();

		int tempnum = (int)(resultdays[num - 1] - lastday + 0.5);
		if (tempnum < 0)
			tempnum = 0;
		resultdays.clear();
		flavorlist[i].num = tempnum;
		cout << flavorlist[i].name << ' ' << flavorlist[i].num << endl;
		for (unsigned int j = 0; j < flavortopre.size(); j++) {
			if (flavortopre[j].name == flavorlist[i].name)
				flavortopre[j] = flavorlist[i];
		}
	}

	vector<Flavor> flavortopre1 = flavortopre;
	vector<Flavor> flavortopre2 = flavortopre;

	int isCpu2 = isCpu;

	(void)allocate(flavortopre1, serverlist1);					//分配过程
	Server::counts = 0;
	isCpu = 0 - isCpu;
	(void)allocate(flavortopre2, serverlist2);
	if (serverlist1.size() <= serverlist2.size())
		serverlist = serverlist1;
	else {
		serverlist = serverlist2;
	}

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



	for (unsigned int i = 0; i < serverlist.size(); i++) {					//打印分配信息
		int cpupro = 0;
		int memorypro = 0;
		cout << "Serverid" << serverlist[i].id << ": ";
		for (int j = 0; j < Prenum; j++) {
			cout << serverlist[i].flavornum[j] << ' ';
			cpupro += serverlist[i].flavornum[j] * flavorlist[j].cpu;
			memorypro += serverlist[i].flavornum[j] * flavorlist[j].memory;
		}
		cout << endl;
		cout << "cpu: " << static_cast<float>(cpupro / (float)Cpu) * 100 << "% memory: " << static_cast<float>(memorypro / (float)Memory) * 100 << "% " << endl;
	}


	// 需要输出的内容
	char * result_file = (char *)finalinfo.c_str();

	// 直接调用输出文件的方法输出到指定文件中(ps请注意格式的正确性，如果有解，第一行只有一个数据；第二行为空；第三行开始才是具体的数据，数据之间用一个空格分隔开)
	write_result(result_file, filename);
}
