#include "MyMatrix.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

//名前空間の省略
using namespace std;

//実際の結果を表示する場合コメントを外す
//#define PRINT_VAL

//要素数
static const int N = 1000;
//ヤコビ法の最大反復回数
static const int MAX = 1000;
//ヤコビ法の許容誤差
static const double EPS = 1.0e-5;

//プロトタイプ宣言
Vector jacobi(Matrix&, Vector&);
Vector gauss(Matrix&, Vector&);

int main()
{
	//ベクトルbの作成
	Vector b(N);
	b.createRand();

	//行列Aの作成
	Matrix A(N, N);
	//密行列で初期化
	A.createDenseMtx();

	//ガウスの消去法で解を求める
	gauss(A, b);

	//cout << A[N-1][0] << endl;

	cout << endl;

	//疎行列で初期化
	A.createSparseMtx(-0.25);


	

	//ヤコビ法で解を求める
	jacobi(A, b);
	//cout << A[N - 1][0] << endl;

	return 0;
}

/*
*以下の問部分を実装していく
*/

//ヤコビ法
Vector jacobi(Matrix& a, Vector& b)
{
	cout << "ヤコビ法" << endl;

	//求めたい未知数ベクトルx
	Vector x(N);

	//任意で値を設定した初期x
	Vector old_x(N);
	const int n = old_x.getSize();
	for(int i = 0; i < n; ++i)
	{
		old_x[i] = 1.0;
	}

	//誤差値の計算結果格納用変数
	double err;

	//反復回数のカウント用変数
	int cnt = 0;

	//時間の計測開始
	clock_t s = clock();

	//最大反復回数になるまで繰り返す
	while(cnt < MAX)
	{
		//解で初期化する
		x = b;

		for(int i = 0; i < N; ++i)
		{
			for(int j = 0; j < N; ++j)
			{
				//問１
				//b-(L+U)*xを行う
				//つまり対角要素以外の時に
				//a * old_xを求め
				//xから減算すればよい
				if (!i==j){
					x[i] -= a[i][j] * old_x[j];
				}
			}
			//問2
			//計算したxを対角要素で除算する
			x[i] /= a[i][i];
		}

		//誤差変数の初期化
		err = 0.0;
		for(int i = 0; i < N; ++i)
		{
			//問3
			//求めたxと、計算前のold_xとの差の総和をerrに入れる
			err += fabs(x[i] - old_x[i]);
		}

		//誤差値が許容範囲になったか
		if(err <= EPS)
		{
			break;
		}

		//求めたxを次の反復計算に使うold_xに代入する
		old_x = x;

		//反復回数のカウント
		++cnt;
	}

	//時間の計測終了
	clock_t e = clock();

	//各種結果の表示
	cout << scientific << "相対誤差:" << (a * x - b).norm() / b.norm() << endl;
	cout << scientific << "絶対誤差：" << err << endl;
	cout << "反復回数：" << cnt << endl;
	cout << "時間：" << e - s << "[ms]" << endl;

	//実際の値の表示
#ifdef PRINT_VAL
	cout << "A =" << endl;
	cout << a << endl;
	cout << "x =" << endl;
	cout << x << endl;
	cout << "b =" << endl;
	cout << b << endl;
	cout << "A * x =" << endl;
	cout << a * x << endl;
#endif

	//結果を返す
	return x;
}

//ガウスの消去法
Vector gauss(Matrix& a_, Vector& b_)
{
	cout << "ガウスの消去法" << endl;

	//計算に必要な変数
	double w;
	//求めたい未知数ベクトルx
	Vector x = b_;
	//行列A
	Matrix a = a_;

	//時間の計測開始
	clock_t s = clock();

	//全ての対角に行うまで繰り返す
	for(int i = 0; i < N - 1; ++i)
	{
		//現在のiをピボットにする
		int p = i;

		//iの対角要素を最大値と仮定する
		double max = fabs(a[i][i]);

		//違う行の同列要素の方が大きいか確認
		for (int j = i + 1; j < N; ++j){
			//問1
			//j行i列目の要素が、現在の最大値maxより大きければ
			//現在のピボットpをjに変更
			//最大値maxを現在見ているj行i番目にする
			if (max < fabs(a[j][i])) {
				max = fabs(a[j][i]);
				p = j;
			}

		}

		//もし最初に選んだのと違う場所が選択されていたら
		if(p != i)
		{
			//問2
			//p行とi行の要素を全て入れ替える
			for(int j = i; j < N; ++j)
			{
				swap(a[p][j], a[i][j]);
				swap(x[i], x[p]);	//ベクトルxの交換も行う理由を考察に書く
			}
			//ベクトルxのi番、p番目も入れ替える
		}

		//前進消去
		for(int j = i + 1; j < N; ++j)
		{
			//問3
			//現在見ている行の対角要素でj行i列目の要素を除算しwに代入する
			//その後、前進消去で同じ列の要素は0になるため
			//j行i列目の要素を0にする
			w = a[j][i] / a[i][i];
			a[j][i] = 0.0;
			//求めたwとi行k列目の要素を乗算したものでj行k列目の要素を減算する
			for (int k = i + 1; k < N; ++k)
			{
				a[j][k] -= w * a[i][k];
			}

			//同様に、ベクトルxのj番目もi番目とwを乗算したもので減算する
			x[j] -= x[i] * w;
		}
	}

	//後退代入
	for(int i = N - 1; i >= 0; --i)
	{
		//問4
		//対角以外の要素を式から無くしたいので
		//i行j列目の要素を対応するj番目のxと乗算したものを
		//対角要素に対応したxのi番目に減算する
		for(int j = i + 1; j < N; j++)
		{
			x[i] -= a[i][j] * x[j];

		}
		x[i] /= a[i][i];
		//行列の対角要素でxの対応した要素を除算する
	}

	//時間の計測終了
	clock_t e = clock();

	//各種結果の表示
	cout << scientific << "相対誤差:" << (a_ * x - b_).norm() / b_.norm() << endl;
	cout << "時間：" << e - s << "[ms]" << endl;

	//値の表示
#ifdef PRINT_VAL
	cout << "A =" << endl;
	cout << a_ << endl;
	cout << "x =" << endl;
	cout << x << endl;
	cout << "b =" << endl;
	cout << b_ << endl;
	cout << "A * x =" << endl;
	cout << a_ * x << endl;
#endif

	//結果を返す
	return x;
}

