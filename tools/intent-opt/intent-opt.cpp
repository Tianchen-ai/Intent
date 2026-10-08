#include "Intent/Compiler/Registration.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  intent::compiler::registerPasses();
  mlir::DialectRegistry registry;
  intent::compiler::registerDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Intent compiler optimizer\n", registry));
}
