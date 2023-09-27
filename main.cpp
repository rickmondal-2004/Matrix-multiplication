#include <oneapi/dpcpp/dpcpp.h>

using namespace std;

int main() {
  // Create a DPC++ queue.
  dpcpp::queue queue;

  // Create two matrices of integers.
  int matrix_a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  int matrix_b[3][3] = {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}};

  // Create a DPC++ buffer for each of the input matrices and the output matrix.
  dpcpp::buffer<int> a_d(matrix_a, sizeof(matrix_a), queue);
  dpcpp::buffer<int> b_d(matrix_b, sizeof(matrix_b), queue);
  dpcpp::buffer<int> c_d(3 * 3, queue);

  // Launch a kernel to perform matrix multiplication.
  queue.submit([&](dpcpp::handler& h) {
    auto a_ptr = a_d.get_ptr();
    auto b_ptr = b_d.get_ptr();
    auto c_ptr = c_d.get_ptr();

    // Iterate over the rows of the output matrix.
    for (int i = 0; i < 3; i++) {
      // Iterate over the columns of the output matrix.
      for (int j = 0; j < 3; j++) {
        // Calculate the element at row i, column j of the output matrix.
        int c_ij = 0;
        for (int k = 0; k < 3; k++) {
          c_ij += a_ptr[i * 3 + k] * b_ptr[k * 3 + j];
        }

        // Store the calculated element in the output matrix.
        c_ptr[i * 3 + j] = c_ij;
      }
    }
  });

  // Wait for the kernel to finish executing.
  queue.wait();

  // Copy the results from the DPC++ buffer to a C++ array.
  int matrix_c[3][3];
  c_d.memcpy(matrix_c, sizeof(matrix_c), queue);

  // Print the results to the console.
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cout << matrix_c[i][j] << " ";
    }
    cout << endl;
  }

  return 0;
}
