R"mlu(
__mlu_func__ unsigned high_product(unsigned x, unsigned y) {
  unsigned a=x>>16,b=x&65535,c=y>>16,d=y&65535;
  unsigned cross=a*d+((b*d)>>16);
  return a*c+(cross>>16)+(((cross&65535)+b*c)>>16);
}
__mlu_func__ float uniform_at(unsigned long long seed, unsigned long long index) {
  const unsigned long long counter = index >> 2;
  unsigned a=counter, b=counter>>32, c=0, d=0;
  unsigned k0=seed, k1=seed>>32;
  for (int round=0;round<10;++round) {
    unsigned left=a*0xd2511f53U, right=c*0xcd9e8d57U;
    unsigned ah=high_product(a,0xd2511f53U), ch=high_product(c,0xcd9e8d57U);
    a=ch^b^k0;
    c=ah^d^k1;
    b=right;
    d=left;
    k0+=0x9e3779b9U;
    k1+=0xbb67ae85U;
  }
  unsigned word=(index&2)?((index&1)?d:c):((index&1)?b:a);
  return (float)(word>>8)*0x1p-24f;
}
)mlu"
