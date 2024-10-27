#ifndef DATASET_H
#define DATASET_H
#include "tensor/xtensor_lib.h"
using namespace std;

template<typename DType, typename LType>
class DataLabel{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    DataLabel(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    xt::xarray<DType> getData() const{ return data; }
    xt::xarray<LType> getLabel() const{ return label; }
};

template<typename DType, typename LType>
class Batch{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    Batch(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    virtual ~Batch(){}
    xt::xarray<DType>& getData(){return data; }
    xt::xarray<LType>& getLabel(){return label; }
};


template<typename DType, typename LType>
class Dataset{
private:
public:
    Dataset(){};
    virtual ~Dataset(){};
    
    virtual int len()=0;
    virtual DataLabel<DType, LType> getitem(int index)=0;
    virtual xt::svector<unsigned long> get_data_shape()=0;
    virtual xt::svector<unsigned long> get_label_shape()=0;
    
};

//////////////////////////////////////////////////////////////////////
template<typename DType, typename LType>
class TensorDataset: public Dataset<DType, LType>
{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;
    
public:
    /* TensorDataset: 
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
    */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label)
    {
        /* TODO: your code is here for the initialization
         */
        this -> data = data;
        this -> label = label;
        this -> data_shape = data.shape();
        this -> label_shape = label.shape();
    }
    /* len():
     *  return the size of dimension 0
    */
    int len()
    {
        /* TODO: your code is here to return the dataset's length
         */
        return data.shape()[0]; //remove it when complete
    }
    
    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index)
    {
        /* TODO: your code is here
         */
        if(index < 0 || index >= len()) throw std::out_of_range("Index is out of range!");

        xt::xarray<DType> single_data = xt::view(data, index, xt::all());
        xt::xarray<LType> single_label;

        if(label.shape().size() > 0)    
        {
            single_label = xt::view(label, index, xt::all());
        }
        else single_label = label;
        
        return DataLabel<DType, LType>(single_data, single_label);
    }
    
    xt::svector<unsigned long> get_data_shape()
    {
        /* TODO: your code is here to return data_shape
         */
        return data_shape;
    }
    xt::svector<unsigned long> get_label_shape()
    {
        /* TODO: your code is here to return label_shape
         */
        return label_shape;
    }
};

#endif /* DATASET_H */
