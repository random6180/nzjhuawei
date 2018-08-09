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



class Matrix
{
public:
	Matrix(); //default constructor
	Matrix(int m, int n);//declare an mxn matrix
	Matrix(const Matrix& A); //copy constructor
	Matrix(const vector<vector<double> >& rhs);

	~Matrix();//destructor
			  //Assignment
	Matrix& operator = (const Matrix& A); //overloading =
										  //operators
	bool operator == (const Matrix& A);//overloading ==

	Matrix& operator += (const Matrix& A); //overloading +=
	Matrix& operator -=(const Matrix& A); //overloading -=
	Matrix& operator *=(const Matrix& A); //overloading *=
	Matrix& operator *=(double a); //overloading *=

	double & operator ()(int i, int j);// access (i,j)
	double & operator()(int i, int j) const; //read only
	Matrix Transpose(); //transpose
	Matrix Inverse();//Inverse Matrix
	bool LU(Matrix& L, Matrix& U);//LU decomposition. return true if successful
	double det();//determinant(Matrix)
	Matrix Adjugate(); //Adjoint/Adjugate
	double Cofactor(int i, int j); //cofactor Cij
	Matrix Cofactor();//matrix of cofactors
	double Minor(int i, int j);//Minor Mij
	const int GetRows() const; //Get the dimension of the Matrix
	const int GetColumns() const;

private:
	int rows;
	int columns;
	double* buf;

};

Matrix operator + (const Matrix& A, const Matrix& B); //Matrix A+B, using += .....
Matrix operator - (const Matrix& A, const Matrix& B); //Matrix A+B, using -= .....
Matrix operator * (const Matrix& A, const Matrix& B); //Matrix A+B, using *= .....
Matrix operator * (double a, const Matrix& A); //Matrix a*A, using *= .....
Matrix operator * (const Matrix& A, double a); //Matrix A*a, using *= .....


											   //------------------------------ Implemenation  -----------------------------
Matrix::Matrix() //default constructor
{
	rows = columns = 0;
	buf = NULL;
	//	cout << "call constructor" << endl;
}
Matrix::Matrix(int m, int n)//declare an mxn matrix
{
	if (!this->buf)delete[] this->buf;
	this->rows = m;
	this->columns = n;
	this->buf = new double[m*n];
	//	cout << "call constructor" << endl;
}
Matrix::Matrix(const Matrix& A) //copy constructor
{
	//                          Matrix C(C); ??????????? could not pass compile, don't worry

	//      if(!this->buf)delete[] this->buf;
	this->rows = A.rows;
	this->columns = A.columns;
	this->buf = new double[(A.rows)*(A.columns)];
	int i;
	for (i = 0; i<((A.rows)*(A.columns)); i++)
	{
		this->buf[i] = A.buf[i];
	}
	//	cout << "call copy construcotor" << endl;
}

Matrix::Matrix(const vector<vector<double> >& rhs) {
	this->rows = rhs.size();
	this->columns = rhs[0].size();
	this->buf = new double[(rhs.size())*(rhs[0].size())];
	for (unsigned int i = 0; i < rhs.size(); i++) {
		for (unsigned int j = 0; j < rhs[0].size(); j++)
			this->buf[i*rhs[0].size() + j] = rhs[i][j];
	}
}


Matrix::~Matrix()//destructor
{
	delete[] this->buf;
	this->rows = 0;
	this->columns = 0;
	//	cout << "call destructor" << endl;
}

Matrix& Matrix::operator = (const Matrix& A) //overloading =
{
	if (this == &A) return *this;
	if (!buf) delete[] buf;
	columns = A.columns;
	rows = A.rows;
	buf = new double[columns*rows];
	int i;
	for (i = 0; i<(columns*rows); i++)
	{
		buf[i] = A.buf[i];
	}
	return *this;
}

bool Matrix::operator == (const Matrix& A)//overloading ==
{
	int i;
	if (!this->buf || !A.buf)
	{
		cout << "Two Empty Matrix" << endl;
		return true;
	}
	if (this->columns != A.columns || this->rows != A.rows)
		return false;
	else
	{
		for (i = 0; i<columns*rows; i++)
		{
			if (abs(this->buf[i] - A.buf[i])>1e-10) return false;
		}

	}
	return true;
}

Matrix& Matrix::operator += (const Matrix& A) //overloading +=
{
	if (!A.buf) return *this;
	if ((this->rows != A.rows) || (this->columns != A.columns))
	{
		//cerr << "Size mismatch in matrix addition" << endl;
		throw logic_error("Size mismatch in matrix addition");
	}
	for (int i = 0; i<A.columns*A.rows; i++)
		this->buf[i] += A.buf[i];

	return *this;
}
Matrix& Matrix::operator -=(const Matrix& A) //overloading -=
{
	if (!A.buf) return *this;
	if ((this->rows != A.rows) || (this->columns != A.columns))
	{
		//cerr << "Size mismatch in matrix addition" << endl;
		throw logic_error("Size mismatch in matrix addition");
	}
	for (int i = 0; i<A.columns*A.rows; i++)
		this->buf[i] -= A.buf[i];
	return *this;
}
Matrix& Matrix::operator *=(const Matrix& A) //overloading *=
{
	if (!A.buf)    throw logic_error(" You are Multipling Empty Matrix");
	if (this->columns != A.rows)    throw logic_error("Size Mismatch!");
	if (A.columns == 0 || A.rows == 0 || this->columns == 0 || this->rows == 0)  throw logic_error("go check your weried matrix first");
	// Matrix tmp(*this);
	//delete[] this->buf;
	//this->buf= new double[this->rows*A.columns];
	Matrix tmp(this->rows, A.columns);
	for (int i = 1; i <= tmp.rows; i++)
	{
		for (int j = 1; j <= tmp.columns; j++)
		{
			tmp(i, j) = 0;
			for (int k = 1; k <= A.rows; k++)
				tmp(i, j) += (*this)(i, k) * A(k, j);
		}
	}
	*this = tmp;
	return *this;
}
Matrix& Matrix::operator *=(double a) //overloading *=
{
	if (!this->buf)	throw logic_error("please Check your empty Matrix first");
	for (int i = 0; i<columns*rows; i++)
	{
		this->buf[i] *= a;
	}
	return *this;
}

double& Matrix::operator ()(int i, int j)// access (i,j)
{
	if (i>this->rows || j>this->columns)   throw logic_error("Matrix is not this big");
	if (i <= 0 || j <= 0)	throw logic_error("can not access, your index is wrong");
	return buf[(i - 1)*columns + (j - 1)]; // is this correct? Unsafe
}
double& Matrix::operator()(int i, int j) const //read only
{
	//return buf[i*columns+j]; // is this correct? Unsafe
	return buf[(i - 1)*columns + (j - 1)]; // is this correct? Unsafe

}

Matrix Matrix::Adjugate() //Adjoint/Adjugate
{
	Matrix tmp;
	tmp = this->Cofactor();
	tmp = tmp.Transpose();
	return tmp;
}
double Matrix::Cofactor(int i, int j) //cofactor Cij
{
	double tmp;
	tmp = this->Minor(i, j);
	tmp = pow(-1, (i + j))*tmp;
	//	double tmp;
	return tmp;
}
Matrix Matrix::Cofactor()//matrix of cofactors
{

	if (!this->buf)    throw logic_error(" Empty Matrix ");
	Matrix tmp(this->rows, this->columns);
	for (int i = 1; i <= this->rows; i++)
	{
		for (int j = 1; j <= this->columns; j++)
		{
			tmp(i, j) = this->Cofactor(i, j);
		}
	}
	return tmp;

}
double Matrix::Minor(int i, int j)//Minor Mij
{
	double tmp;
	Matrix A;
	A.rows = (this->rows) - 1;
	A.columns = (this->columns) - 1;
	A.buf = new double[A.rows*A.columns];
	int a = 0;
	for (int m = 1; m <= this->rows; m++)
	{
		for (int n = 1; n <= this->columns; n++)
		{
			if (m == i) continue;
			if (n == j) continue;
			A.buf[a] = (*this)(m, n);
			a++;
		}
	}
	tmp = A.det();
	return tmp;

}

const int Matrix::GetRows() const
{
	return rows;
};

const int Matrix::GetColumns() const
{
	return columns; //
};
Matrix Matrix::Transpose()  //transpose
{
	//check size
	if ((this->GetRows() == 0) || (this->GetColumns() == 0))
	{
		//cerr << "Missing matrix data" << endl;
		throw invalid_argument("Missing matrix data");
	}

	Matrix tmp(this->GetColumns(), this->GetRows());
	for (int i = 1; i <= tmp.GetRows(); ++i)
	{
		for (int j = 1; j <= tmp.GetColumns(); ++j)
			tmp(i, j) = (*this)(j, i);
	}
	return tmp;
}
Matrix Matrix::Inverse()//Inverse Matrix
{
	Matrix tmp;
	if ((*this).GetRows() != (*this).GetColumns())
	{
		throw logic_error("Solving for Inverse fail: Not a square    matrix!");
		//return (*this);
	}
	if (fabs(this->det() - 0)<0.000000001)
	{
		throw logic_error("determinant equal to zero, can not do inverse");
	}
	Matrix A;
	A = this->Adjugate();
	tmp = (1 / this->det())*A;
	return tmp;
}

bool Matrix::LU(Matrix& L, Matrix& U)//LU decomposition. return true if successful
{
	if (this->rows != this->columns)   //LU decomposition only can be operated on square Matrix
	{
		cout << "Matrix A is not a square Matrix! Please check the data again." << endl;
		return false;
	}


	double bigflag = 0.0; // bigflag is used for storing the biggest value date in the matrix A to judge whether all of the items in a rows equals 0( in this case, |A|=0, ;
	for (long i = 0; i< this->rows; i++)  //primary check the prerequisite for LU decomposition.
	{
		bigflag = 0.0;
		for (long j = 0; j<this->columns; j++)   //Becareful A.rows=A.columns! here is just for easy reading and understanding
		{
			//if(fabs(*(A.buf+A.rows*i+j))>bigflag)
			if (fabs(this->buf[i*this->rows + j])>bigflag)

				//bigflag=*(A.buf+A.rows*i+j);    //to find the biggest value data in one row
				bigflag = this->buf[i*this->rows + j];
		}
		if (bigflag == 0.0)
		{
			cout << "No nonzero largest element in the" << i + 1 << "th rows.(The det(A)=0, which does not meet the requestment of LU decomposition.)" << endl;
			//" Matrix A may be a SINGULAR Matrix, which maybe can't be decomposed. Should be check again???"
			return false;
		}

	}

	if (L.buf)  //if L have data, delete the data 
	{
		L.rows = L.columns = 0;
		delete[] L.buf;
		L.buf = NULL;
	}   //End of if L.buf... 
	if (U.buf)  //if U have data, delete the data 
	{
		U.rows = U.columns = 0;
		delete[] U.buf;
		U.buf = NULL;
	}   //End of if U.buf ...         

	L.rows = U.rows = this->rows;
	L.columns = U.columns = this->columns;  //set the L,U's rows and columns the same as the A's
	L.buf = new double[L.rows*L.columns];  //creat a buffer for storing the data 
	U.buf = new double[U.rows*U.columns];
	for (long i = 0; i<L.rows; i++)
		for (long j = 0; j<L.columns; j++)
		{
			*(L.buf + L.columns*i + j) = 0.0;
			*(U.buf + U.columns*i + j) = 0.0;
		}    //init L,U=0;

	double sum = 0.0; //temp varity used for store A[i][j] during the loop but init it to 0.0 here..

					  //	cout << "LU decomposing.";    //if n=very large, then should to tell user I am running.
	for (long n = 0; n<this->rows; n++)  //n used for A[n][n]
	{
		// all L[n][n] is assumed =1.0;
		for (long j = n; j<U.columns; j++)   // for caculating U[][j], the nth rows
		{
			sum = *(this->buf + n*this->rows + j);    //here sum store A[n][j] 
			for (long k = 0; k<n; k++)
				sum -= (*(L.buf + n*L.columns + k))*(*(U.buf + k*U.columns + j));
			*(U.buf + n*U.columns + j) = sum;
		}

		for (long i = 0; i<L.rows; i++)   //for caculating L[i][], the nth columns    
		{
			sum = *(this->buf + i*this->rows + n);    //here sum store A[n][j]
			for (long k = 0; k<n; k++)
				sum -= (*(L.buf + i*L.rows + k))*(*(U.buf + k*U.rows + n));
			if (*(U.buf + n*U.rows + n) == 0)
			{   //if U[n][n]==0, then it means that A can not decomposed, for if U[n][n]=0, |A|=|L|*|U|=|L[n][n]|*|U[n][n]|=0, which can not meet the prerequisite of decomposing A;
				cout << "OOps, Zero in U[" << n << "][" << n << "] is found, the matrix can not be decomposed. " << endl;
				L.rows = L.columns = 0;
				U.rows = U.columns = 0;
				delete[] L.buf;
				L.buf = NULL;
				delete[] U.buf;
				U.buf = NULL; //Clear all of the value is useful for other function who call LU(), and detect whether the operation of LU is successful.Also free all the memory.
				return false;
			}
			else
			{
				*(L.buf + i*L.rows + n) = sum*1.0 / (*(U.buf + n*U.rows + n));
			}
		}
		//		cout << ".";
	}

	return true;

}

double Matrix::det()//determinant(Matrix)
{
	if (this->rows != this->columns)
		throw logic_error("Matrix has to be square to find det");
	int r = this->GetRows();
	double tmp = 1;
	Matrix L, U;
	this->LU(L, U);
	for (int i = 0; i<r; i++)
		tmp = tmp*L.buf[i*r + i] * U.buf[i*r + i];
	return tmp;



}


Matrix operator + (const Matrix& A, const Matrix& B) //Matrix A+B, using += .....
{
	Matrix tmp = A;
	tmp += B;//use "+="
	return tmp;//   done
}

Matrix operator - (const Matrix& A, const Matrix& B) //Matrix A+B, using -= .....
{
	Matrix tmp = A;
	tmp -= B;//use "-="
	return tmp;//    done
}

Matrix operator * (const Matrix& A, const Matrix& B) //Matrix A+B, using *= .....
{
	Matrix tmp = A;
	tmp *= B;//use "*="
	return tmp;//    done
}


Matrix operator * (double a, const Matrix& A) //Matrix a*A, using *= .....
{
	Matrix tmp = A;
	//do a*A
	tmp *= a;
	return tmp;   //done
};


Matrix operator * (const Matrix& A, double a) //Matrix A*a, using *= .....
{
	Matrix tmp = A;
	//do A*a
	tmp *= a;
	return tmp;
};


double diffSquareSum(vector<double>& op1, vector<double>& op2) {	//两个数组差的平方和，即距离的平方
	double sum = 0;
	for (unsigned int i = 0; i < op1.size(); i++)
		sum += (op1[i] - op2[i])*(op1[i] - op2[i]);
	return sum;
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
float simCalculate(Flavor& flavor, Server& server) {
	float tmp = (float)((Memory - server.memory + flavor.memory) / (Cpu - server.cpu + flavor.cpu));
	return abs(tmp - (Memory / Cpu));
}

int selectFlavor(vector<Flavor>& flavorlist, Server& server, int last, int first) {    //选出合适的k
	float minmum = 5000;
	float maxmum = 1000;
	float minmum2 = 5000;
	int k = last;

	if (last < first)
		return -1;

	for (int i = last; i >= first; i--) {		//寻找k
		if (flavorlist[i].num > 0) {
			if (server.cpu == Cpu || isCpu * (float)((Memory - server.memory) / (Cpu - server.cpu)) < isCpu*(float)(Memory / Cpu)) {		//最不平衡
				float tmpmax = (float)(flavorlist[i].memory / flavorlist[i].cpu);
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
				float tmpmin = simCalculate(flavorlist[i], server);
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



void exceptionDetect(vector<double>& input, double rate = 3, double dif = 2) {	//rate为倍率，即高于平均值或低于平均值多少倍，则判为异常点
	if (input.size() <= 1)
		return;
	double sum = 0;
	double avg = 0.0;
	for (unsigned int i = 0; i < input.size(); i++) {
		sum += input[i];
	}
	for (unsigned int j = 0; j < input.size(); j++) {
		avg = (double)((sum - input[j]) / (input.size() - 1));
		if (avg == 0)
			return;
		if (input[j] / avg > rate && input[j] - avg > dif) {
			//			cout << input[j] << ":" << avg << endl;
			input[j] = avg;
		}
	}
}


//一元非线性回归
vector<double> generatetheta(vector<double>& xArr, vector<double>& yArr, double xp, int order = 10, int kp=5) {//x为预测点
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
		//		w = exp(-(xp - xArr[n])*(xp - xArr[n]));
		//		cout << "(xp - xArr[n])*(xp - xArr[n]):" << (xp - xArr[n])*(xp - xArr[n]) << endl;
		//		cout << "(xp - xArr[n])*(xp - xArr[n]) / -2 * kp*kp:" << (xp - xArr[n])*(xp - xArr[n]) / -2 * kp*kp << endl;
				w = exp((xp - xArr[n])*(xp - xArr[n]) /(-2 * kp * kp));
		//		w = 1;
		//		cout << "w:" << w << endl;
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
	A = X.Inverse() * Y;
	vector<double> theta;
	for (int t = 0; t < A.GetRows(); t++) {
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
	

	vector<int> resultdays;
	double result=0;

	//vector<double> xArrx;
	//for (unsigned int u = 0; u < dayvector.size(); u++) {
	//	xArrx.push_back(dayvector[u]);
	//}
	//sort(xArrx.begin(), xArrx.end());
	//double min = xArrx[0];
	//double max = xArrx[xArrx.size() - 1];
	//for (unsigned int y = 0; y < dayvector.size(); y++) {
	//	dayvector[y] = dayvector[y] / (max - min) - min / (max - min) + 1;			//归[1,2]
	//	cout << dayvector[y] << endl;
	//}
	



	for (unsigned int i = 0; i < flavorlist.size(); i++) {									//对flavor对象的num赋值，即预测过程
		for (int ui = 0; ui < days; ui++) {
			dayvector.push_back((double)(ui + 1));			//生成从1开始的天数
		}


	//	cout << "flavor:" << i + 1 << endl;

		exceptionDetect(floatalllist[i], 5);
		sumflavorlist = sumflavor(floatalllist[i]);
		double lastday = sumflavorlist[sumflavorlist.size() - 1];
//		cout << "lastday:" << lastday << endl;
//		theta = generatetheta(dayvector, sumflavorlist, 3);

//		theta = generatetheta(dayvector, floatalllist[i], 4);

		for (int daynum = 0; daynum < num; daynum++) {

			double xp = floatalllist[0].size() + daynum + 1;
			theta = generatetheta(dayvector, sumflavorlist, xp, 1, 10);
			result = preOutput(theta, xp);
//			cout << "flavor" << i + 1 << ' ' << "result:" << result << endl;
			resultdays.push_back((int)(result));

		}

		for (int daynum = 0; daynum < num; daynum++) {
			double xp = floatalllist[0].size() + daynum + 1;
			sumflavorlist.push_back(resultdays[daynum]);
			dayvector.push_back(xp);
		}

		resultdays.clear();
		for (int daynum = 0; daynum < num; daynum++) {

			double xp = floatalllist[0].size() + daynum + 1;
			theta = generatetheta(dayvector, sumflavorlist, xp, 3, 100);
			result = preOutput(theta, xp);
			resultdays.push_back((int)(result));

		}

		dayvector.clear();

		

		int tempnum = (int)(resultdays[num - 1] - lastday);
	//	int tempnum = sum(resultdays);
		if (tempnum < 0)
			tempnum = 0;
		resultdays.clear();
		flavorlist[i].num = tempnum;
//		cout << flavorlist[i].name << ' ' << flavorlist[i].num << endl;
		for (unsigned int j = 0; j < flavortopre.size(); j++) {
			if (flavortopre[j].name == flavorlist[i].name)
				flavortopre[j] = flavorlist[i];
		}
	}

	flavortopre[0].num = 44;
	flavortopre[1].num = 0;
	flavortopre[2].num = 121;
	flavortopre[3].num = 0;
	flavortopre[4].num = 22;
	flavortopre[5].num = 0;
	flavortopre[6].num = 187;
	flavortopre[7].num = 0;
	flavortopre[8].num = 17;
	flavortopre[9].num = 0;
	flavortopre[10].num = 41;
	flavortopre[11].num = 0;
	flavortopre[12].num = 67;
	flavortopre[13].num = 0;
	flavortopre[14].num = 193;

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
	vector<Flavor> flavortopre1 = flavortopre;
	vector<Flavor> flavortopre2 = flavortopre;


	(void)allocate(flavortopre1, serverlist1);					//分配过程
	Server::counts = 0;
	isCpu = 0 - isCpu;
	(void)allocate(flavortopre2, serverlist2);
	if (serverlist1.size() <= serverlist2.size())
		serverlist = serverlist1;
	else {
		serverlist = serverlist2;
		cout << "change.........." << endl;
	}
//	serverlist = serverlist1;

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
