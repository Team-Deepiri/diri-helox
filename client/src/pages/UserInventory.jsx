import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-hot-toast';
import { 
  PlusIcon, 
  MagnifyingGlassIcon, 
  FunnelIcon,
  Squares2X2Icon,
  ListBulletIcon,
  ArrowDownTrayIcon,
  ShareIcon,
  HeartIcon,
  SparklesIcon,
  TagIcon
} from '@heroicons/react/24/outline';
import { userItemsApi, ITEM_CATEGORIES, RARITY_LEVELS } from '../api/userItemsApi';
import ItemCard from '../components/ItemCard';
import AddItemModal from '../components/AddItemModal';
import ItemDetailModal from '../components/ItemDetailModal';

const UserInventory = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedRarity, setSelectedRarity] = useState('all');
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'list'
  const [sortBy, setSortBy] = useState('createdAt');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [selectedItem, setSelectedItem] = useState(null);
  const [showFilters, setShowFilters] = useState(false);

  const queryClient = useQueryClient();

  // Build query options
  const queryOptions = {
    category: selectedCategory,
    isFavorite: showFavoritesOnly || undefined,
    sort: sortBy,
    order: sortOrder,
    limit: 24,
    page: currentPage
  };

  if (selectedRarity !== 'all') {
    queryOptions.rarity = selectedRarity;
  }

  // Fetch items
  const { 
    data: itemsResponse, 
    isLoading, 
    error 
  } = useQuery(
    ['userItems', queryOptions, searchQuery],
    () => searchQuery 
      ? userItemsApi.searchItems(searchQuery, queryOptions)
      : userItemsApi.getItems(queryOptions),
    {
      keepPreviousData: true,
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  );

  // Fetch stats
  const { data: statsResponse } = useQuery(
    ['userItemStats'],
    userItemsApi.getStats,
    {
      staleTime: 10 * 60 * 1000, // 10 minutes
    }
  );

  // Mutations
  const createItemMutation = useMutation(userItemsApi.createItem, {
    onSuccess: () => {
      queryClient.invalidateQueries(['userItems']);
      queryClient.invalidateQueries(['userItemStats']);
      toast.success('Item added successfully!');
      setShowAddModal(false);
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to add item');
    }
  });

  const updateItemMutation = useMutation(
    ({ itemId, data }) => userItemsApi.updateItem(itemId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['userItems']);
        queryClient.invalidateQueries(['userItemStats']);
        toast.success('Item updated successfully!');
        setShowAddModal(false);
      },
      onError: (error) => {
        toast.error(error.message || 'Failed to update item');
      }
    }
  );

  const toggleFavoriteMutation = useMutation(userItemsApi.toggleFavorite, {
    onSuccess: () => {
      queryClient.invalidateQueries(['userItems']);
      queryClient.invalidateQueries(['userItemStats']);
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to update favorite status');
    }
  });

  const deleteItemMutation = useMutation(
    ({ itemId, permanent }) => userItemsApi.deleteItem(itemId, permanent),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['userItems']);
        queryClient.invalidateQueries(['userItemStats']);
        toast.success('Item deleted successfully!');
        setShowDetailModal(false);
      },
      onError: (error) => {
        toast.error(error.message || 'Failed to delete item');
      }
    }
  );

  const addMemoryMutation = useMutation(
    ({ itemId, memoryData }) => userItemsApi.addMemory(itemId, memoryData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['userItems']);
        toast.success('Memory added successfully!');
      },
      onError: (error) => {
        toast.error(error.message || 'Failed to add memory');
      }
    }
  );

  // Event handlers
  const handleAddItem = (itemData) => {
    createItemMutation.mutate(itemData);
  };

  const handleEditItem = (item) => {
    setSelectedItem(item);
    setShowAddModal(true);
  };

  const handleUpdateItem = (itemData) => {
    if (selectedItem) {
      updateItemMutation.mutate({ itemId: selectedItem._id, data: itemData });
    }
  };

  const handleViewItem = (item) => {
    setSelectedItem(item);
    setShowDetailModal(true);
  };

  const handleDeleteItem = (item) => {
    if (window.confirm('Are you sure you want to delete this item?')) {
      deleteItemMutation.mutate({ itemId: item._id, permanent: false });
    }
  };

  const handleToggleFavorite = (itemId) => {
    toggleFavoriteMutation.mutate(itemId);
  };

  const handleAddMemory = (itemId, memoryData) => {
    addMemoryMutation.mutate({ itemId, memoryData });
  };

  const handleShare = (item) => {
    // Simple share functionality - copy link to clipboard
    const shareText = `Check out my ${item.name} from my adventure collection!`;
    if (navigator.share) {
      navigator.share({
        title: item.name,
        text: shareText,
        url: window.location.href
      });
    } else {
      navigator.clipboard.writeText(shareText);
      toast.success('Share text copied to clipboard!');
    }
  };

  const handleExport = async () => {
    try {
      const data = await userItemsApi.exportItems('json');
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'my-inventory.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('Inventory exported successfully!');
    } catch (error) {
      toast.error('Failed to export inventory');
    }
  };

  const resetFilters = () => {
    setSearchQuery('');
    setSelectedCategory('all');
    setSelectedRarity('all');
    setShowFavoritesOnly(false);
    setCurrentPage(1);
  };

  const items = itemsResponse?.data || [];
  const stats = statsResponse?.data || {};
  const totalItems = stats.totalItems || 0;
  const totalValue = stats.totalValue || 0;
  const favoriteCount = stats.favoriteCount || 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-white">My Inventory</h1>
          <p className="text-gray-300 text-lg">
            Manage your adventure collection and memories
          </p>
          
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold text-white">{totalItems}</div>
              <div className="text-gray-300">Total Items</div>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold text-yellow-400">{totalValue}</div>
              <div className="text-gray-300">Total Points</div>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold text-red-400">{favoriteCount}</div>
              <div className="text-gray-300">Favorites</div>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
              <div className="text-3xl font-bold text-purple-400">
                {stats.categoryStats?.length || 0}
              </div>
              <div className="text-gray-300">Categories</div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-6">
          <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                placeholder="Search items..."
              />
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`
                  px-4 py-2 rounded-lg border transition-colors flex items-center gap-2
                  ${showFilters 
                    ? 'bg-purple-600 border-purple-600 text-white' 
                    : 'bg-white/10 border-white/20 text-gray-300 hover:bg-white/20'
                  }
                `}
              >
                <FunnelIcon className="w-4 h-4" />
                Filters
              </button>

              <div className="flex items-center bg-white/10 rounded-lg border border-white/20">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`
                    p-2 rounded-l-lg transition-colors
                    ${viewMode === 'grid' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}
                  `}
                >
                  <Squares2X2Icon className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`
                    p-2 rounded-r-lg transition-colors
                    ${viewMode === 'list' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'}
                  `}
                >
                  <ListBulletIcon className="w-5 h-5" />
                </button>
              </div>

              <button
                onClick={handleExport}
                className="px-4 py-2 bg-white/10 border border-white/20 text-gray-300 hover:bg-white/20 rounded-lg transition-colors flex items-center gap-2"
              >
                <ArrowDownTrayIcon className="w-4 h-4" />
                Export
              </button>

              <button
                onClick={() => {
                  setSelectedItem(null);
                  setShowAddModal(true);
                }}
                className="px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2"
              >
                <PlusIcon className="w-4 h-4" />
                Add Item
              </button>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <div className="mt-6 pt-6 border-t border-white/10 grid grid-cols-1 md:grid-cols-4 gap-4">
              {/* Category Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Category</label>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="all" className="bg-gray-800">All Categories</option>
                  {ITEM_CATEGORIES.map(category => (
                    <option key={category.value} value={category.value} className="bg-gray-800">
                      {category.icon} {category.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Rarity Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Rarity</label>
                <select
                  value={selectedRarity}
                  onChange={(e) => setSelectedRarity(e.target.value)}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="all" className="bg-gray-800">All Rarities</option>
                  {RARITY_LEVELS.map(rarity => (
                    <option key={rarity.value} value={rarity.value} className="bg-gray-800">
                      {rarity.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Sort */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Sort By</label>
                <select
                  value={`${sortBy}-${sortOrder}`}
                  onChange={(e) => {
                    const [field, order] = e.target.value.split('-');
                    setSortBy(field);
                    setSortOrder(order);
                  }}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="createdAt-desc" className="bg-gray-800">Newest First</option>
                  <option value="createdAt-asc" className="bg-gray-800">Oldest First</option>
                  <option value="name-asc" className="bg-gray-800">Name A-Z</option>
                  <option value="name-desc" className="bg-gray-800">Name Z-A</option>
                  <option value="value.points-desc" className="bg-gray-800">Highest Value</option>
                  <option value="value.points-asc" className="bg-gray-800">Lowest Value</option>
                </select>
              </div>

              {/* Favorites Toggle & Reset */}
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showFavoritesOnly}
                    onChange={(e) => setShowFavoritesOnly(e.target.checked)}
                    className="w-4 h-4 text-purple-600 bg-white/10 border-white/20 rounded focus:ring-purple-500"
                  />
                  <span className="text-gray-300 text-sm">Favorites Only</span>
                </label>
                <button
                  onClick={resetFilters}
                  className="w-full px-3 py-2 text-sm bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Reset Filters
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Items Grid/List */}
        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <div className="text-red-400 mb-4">Failed to load items</div>
            <button
              onClick={() => queryClient.invalidateQueries(['userItems'])}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📦</div>
            <h3 className="text-xl font-semibold text-white mb-2">No items found</h3>
            <p className="text-gray-400 mb-6">
              {searchQuery || selectedCategory !== 'all' || showFavoritesOnly
                ? 'Try adjusting your filters or search terms'
                : 'Start building your collection by adding your first item!'
              }
            </p>
            <button
              onClick={() => {
                setSelectedItem(null);
                setShowAddModal(true);
              }}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
            >
              Add Your First Item
            </button>
          </div>
        ) : (
          <div className={
            viewMode === 'grid' 
              ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6'
              : 'space-y-4'
          }>
            {items.map(item => (
              <ItemCard
                key={item._id}
                item={item}
                onToggleFavorite={handleToggleFavorite}
                onEdit={handleEditItem}
                onDelete={handleDeleteItem}
                onView={handleViewItem}
                onShare={handleShare}
                className={viewMode === 'list' ? 'flex flex-row h-32' : ''}
              />
            ))}
          </div>
        )}

        {/* Pagination */}
        {items.length > 0 && itemsResponse?.pagination && (
          <div className="flex justify-center items-center gap-4 py-6">
            <button
              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="px-4 py-2 bg-white/10 border border-white/20 text-gray-300 hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
            >
              Previous
            </button>
            
            <span className="text-gray-300">
              Page {currentPage}
            </span>
            
            <button
              onClick={() => setCurrentPage(prev => prev + 1)}
              disabled={items.length < 24}
              className="px-4 py-2 bg-white/10 border border-white/20 text-gray-300 hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Modals */}
      <AddItemModal
        isOpen={showAddModal}
        onClose={() => {
          setShowAddModal(false);
          setSelectedItem(null);
        }}
        onSubmit={selectedItem ? handleUpdateItem : handleAddItem}
        initialData={selectedItem}
      />

      <ItemDetailModal
        item={selectedItem}
        isOpen={showDetailModal}
        onClose={() => {
          setShowDetailModal(false);
          setSelectedItem(null);
        }}
        onEdit={handleEditItem}
        onDelete={handleDeleteItem}
        onToggleFavorite={handleToggleFavorite}
        onAddMemory={handleAddMemory}
        onShare={handleShare}
      />
    </div>
  );
};

export default UserInventory;
