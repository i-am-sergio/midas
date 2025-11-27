# Gold standard for descomposition of monoliths into microservices

## JPetStore
**Microservices:**

1. **Account Service**
- Account
- AccountDao
- SqlMapAccountDao

2. **Catalog Service**
- Category
- CategoryDao
- SqlMapCategoryDao
- Item
- ItemDao
- SqlMapItemDao
- Product
- ProductDao
- SqlMapProductDao

3. **Order Service**
- Order
- OrderDao
- SqlMapOrderDao
- OrderService
- LineItem

4. **Cart Service**
- Cart
- CartItem

5. **Sequence Service**
- Sequence
- SqlMapSequenceDao

6. **API Gateway Service**
- PetStoreFacade
- PetStoreImpl

## AcmeAir
**Microservices:**

1. **Authentication Service**
2. **Customer Service**
3. **Flight Service**
4. **Booking Service**

## Plants
**Microservices:**


## DayTrader
**Microservices:**
