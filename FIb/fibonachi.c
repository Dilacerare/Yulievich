#include <stdio.h>
#include <time.h>
#include <stdlib.h>

unsigned long int fibonacci(int n) 
{
	if (n <= 1) 
	{
		return n;
	} 
	else 
	{
		return fibonacci(n-1) + fibonacci(n-2);
	}
}

int main(int argc, char  *argv[]) {
	int n = atoi(argv[1]);
	//int n, i;
	clock_t start, end;

	//printf("Enter the number of terms: ");
	//scanf("%d", &n);

	start = clock();

	/*printf("Fibonacci Series: ");*/

	/*for (i = 0; i < n; i++) {
	printf("%d ", fibonacci(i));
	}*/

	//printf("Fibonacci Series: %ld\n", fibonacci(n));
	
	fibonacci(n);

	end = clock();

	printf("%d, %.10f\n", n, ((double)(end - start))/CLOCKS_PER_SEC);

	return 0;
}

