#ifndef _DARRAY_H_
#define _DARRAY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <list>

namespace DartLib
{

/*!
 *  A simple dynamic array class.
 *    NOTE: This is designed specially for convex hull algorithm,
 *          please do NOT use it in general case.
 */


template <typename T>
class CDArray
{
public:

    CDArray();
    CDArray(size_t size);
    ~CDArray();

    T operator[](size_t i) const { return m_data[i]; };

    T& operator[](size_t i) { return m_data[i]; };
    
    CDArray<T>& operator=(const CDArray<T>& rhs);

    size_t find(const T& v);

    size_t end() { return m_size; };

    T& back() { return m_data[m_size - 1]; };

    void push(const T& v);

    T& allocate();

    void clear();

    /*!
     *  The index of the newly added element after push or allocate methods.
     */
    size_t new_index() const { return m_new_index; };

    void remove(size_t i);

    void remove(const T& v);

    /*!
     *  The size of the normal elements.
     */
    size_t size() const { return m_size; };
    size_t capacity() const { return m_capacity; };

    bool is_deleted(size_t idx) const { return m_tag[idx] == ETag::REMOVED; };
  protected:
    

  protected:
    enum ETag : char { NORMAL, REMOVED };

    T*     m_data;
    std::list<size_t> m_deleted_items;  // store the indices of the removed items
    char*  m_tag;     // TODO: change to store multi-tags
    size_t m_new_index;

    size_t m_size;      // the size of the all stored elements
    size_t m_capacity;
    size_t m_step_capacity = 4096;
};

template <typename T>
CDArray<T>::CDArray()
{
    m_new_index = 0;
    m_size = 0;
    m_capacity = 4096;
    m_data = (T*) malloc((size_t)(m_capacity * sizeof(T)));
    m_tag  = (char*) malloc((size_t)(m_capacity * sizeof(char)));
}

template <typename T>
CDArray<T>::CDArray(size_t size)
{
    m_new_index = 0;
    m_size = size;
    m_capacity = size;
    m_data = (T*) malloc((size_t)(m_capacity * sizeof(T)));
    m_tag  = (char*) malloc((size_t)(m_capacity * sizeof(char)));
}

template <typename T>
CDArray<T>::~CDArray()
{
    m_new_index = 0;
    m_size = 0;
    m_capacity = 0;
    free(m_data); 
    m_data = NULL;
    free(m_tag);
    m_tag = NULL;
    m_deleted_items.clear();
}

template <typename T>
CDArray<T>& CDArray<T>::operator=(const CDArray<T>& rhs)
{
    m_new_index = rhs.m_new_index;
    m_size = rhs.m_size;
    m_capacity = rhs.m_capacity;
    m_step_capacity = rhs.m_step_capacity;
    
    m_data = (T*) malloc((size_t)(m_capacity * sizeof(T)));
    memcpy(m_data, rhs.m_data, (size_t)(m_size * sizeof(T)));

    m_tag = (char*) malloc((size_t)(m_capacity * sizeof(char)));
    memcpy(m_tag, rhs.m_tag, (size_t)(m_size * sizeof(char)));
    
    for (auto v : rhs.m_deleted_items)
        m_deleted_items.push_back(v);

    return *this;
}

template <typename T>
size_t CDArray<T>::find(const T& v)
{
    for (size_t i = 0; i < m_size; ++i)
    {
        if (m_data[i] == v && m_tag[i] == ETag::NORMAL)
            return i;
    }

    return m_size;
}

template <typename T>
void CDArray<T>::push(const T& v)
{
    T& data = this->allocate();
    data = v;
}

template <typename T>
T& CDArray<T>::allocate()
{
    if (m_deleted_items.size() > 0)
    {
        size_t idx = m_deleted_items.front();
        m_deleted_items.pop_front();
        m_new_index = idx;
        m_tag[idx] = ETag::NORMAL;
        return m_data[idx];
    }

    if (m_size + 1 <= m_capacity)
    {
        m_new_index = m_size;
        m_tag[m_size] = ETag::NORMAL;
        return m_data[m_size++];
    }
    else
    {
        T* data = m_data;
        size_t data_length = (size_t)(m_capacity * sizeof(T));
        char* tag = m_tag;
        size_t tag_length = (size_t)(m_capacity * sizeof(char));

        m_capacity += m_step_capacity;
        m_data = (T*) malloc((size_t)(m_capacity * sizeof(T)));
        m_tag  = (char*) malloc((size_t)(m_capacity * sizeof(char)));

        if (m_data == NULL || m_tag == NULL)
        {
            printf("ERROR: allocate memory space failed!\n");
            exit(EXIT_FAILURE);
        }

        // copy
        memcpy(m_data, data, data_length);
        free(data);

        memcpy(m_tag, tag, tag_length);
        free(tag);

        return allocate();
    }
}

template <typename T>
void CDArray<T>::clear()
{
    m_new_index = 0;
    m_size = 0;
    m_deleted_items.clear();
}

template <typename T>
void CDArray<T>::remove(size_t i)
{
    if (i < m_size)
    {
        m_tag[i] = ETag::REMOVED;
        m_deleted_items.push_back(i);
    }
}

template <typename T>
void CDArray<T>::remove(const T& v)
{
    size_t i = find(v);
    if (i != end()) remove(i);
}

} // namespace DartLib
#endif //! _DARRAY_H_
