data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> Q;
  // matrix[N, J] data_matrix[Q];
  array[Q] complex_matrix[N, J] data_matrix;
  // int data_matrix2[Q, N, J]; // specify the same data with data_matrix
  // vector[N] data_vector[Q];
}

parameters {
  real y;
}

model {
  y ~ normal(0, 1);
  print("dm=", data_matrix); 
  print("dm[1]=", data_matrix[1]);
  print("dm[1, 1,]=", data_matrix[1, 1, ]);
  
  // print("dm2[3,2,1]=", data_matrix2[3,2,1]);
  // print("dm[2,1,2]=", data_matrix[2,1,2]);
  // print("dm2[2,1,2]=", data_matrix[2,1,2]);
  // print("dm2=", data_matrix2); 
  // print("dv=", data_vector);
  print("");
} 