R"mlu(
struct Moment { int n; float mean,m2; };
__mlu_func__ Moment merge_moments(Moment a,Moment b){
  if(!a.n)return b;
  if(!b.n)return a;
  int n=a.n+b.n;
  float delta=b.mean-a.mean;
  float change=__cn_scalar_div_f32_rn(delta*(float)b.n,(float)n);
  return Moment{n,a.mean+change,a.m2+b.m2+delta*(float)a.n*change};
}
)mlu"
