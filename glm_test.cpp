#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
#include <glm/gtx/component_wise.hpp>
#include <iostream>

int main(){
    // glm::mat3 m1{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    // glm::mat3 m2{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    // std::cout << m1 * m2 << std::endl;
    // std::cout << glm::matrixCompMult(m1, m2) << std::endl;

    // glm::vec3 v1{1., 2., 3.};
    // glm::vec3 v2{1., 2., 3.};
    // std::cout << v1 * v2 <<std::endl;

    glm::mat3 m{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    std::cout << m << std::endl;
    std::cout << m[0][1] << ", " << m[1][0] << std::endl;

    glm::vec3 v{1., 2., 3.};
    
    float a = 0.f;
    float b = 1.f;

    float c = a * -b;

    return 0;
}