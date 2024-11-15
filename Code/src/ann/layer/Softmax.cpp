/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   Softmax.cpp
 * Author: ltsach
 * 
 * Created on August 25, 2024, 2:46 PM
 */

#include "layer/Softmax.h"
#include "ann/functions.h"
#include "sformat/fmt_lib.h"
#include <filesystem> //require C++17
namespace fs = std::filesystem;

Softmax::Softmax(int axis, string name): m_nAxis(axis)
{
    if(trim(name).size() != 0) m_sName = name;
    else m_sName = "Softmax_" + to_string(++m_unLayer_idx);
}

Softmax::Softmax(const Softmax& orig){}

Softmax::~Softmax(){}

xt::xarray<double> Softmax::forward(xt::xarray<double> X)
{
    //YOUR CODE IS HERE
    m_aCached_Y = softmax(X, m_nAxis);

    return m_aCached_Y;
}

xt::xarray<double> Softmax::backward(xt::xarray<double> DY)
{
    //YOUR CODE IS HERE
    xt::xarray<double> backpropagation = xt::zeros_like(DY);
    xt::xarray<double> DIAG_Y = diag_stack(m_aCached_Y);
    // xt::xarray<double> m_aCached_Y_T = xt::transpose(m_aCached_Y);
    xt::xarray<double> OUTER_multiplication = outer_stack(m_aCached_Y, m_aCached_Y);
    xt::xarray<double> Jacobian = DIAG_Y - OUTER_multiplication;
    backpropagation = matmul_on_stack(Jacobian, DY);

    return backpropagation;
}

string Softmax::get_desc()
{
    string desc = fmt::format("{:<10s}, {:<15s}: {:4d}",
                    "Softmax", this->getname(), m_nAxis);
    return desc;
}
