import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from 'react-query';
import { 
  ArrowRightIcon, 
  SparklesIcon, 
  HeartIcon,
  PlusIcon,
  EyeIcon
} from '@heroicons/react/24/outline';
import { userItemsApi, RARITY_LEVELS } from '../api/userItemsApi';

const InventoryWidget = () => {
  // Fetch user item stats
  const { data: statsResponse, isLoading: statsLoading } = useQuery(
    ['userItemStats'],
    userItemsApi.getStats,
    {
      staleTime: 10 * 60 * 1000, // 10 minutes
    }
  );

  // Fetch recent items
  const { data: recentItemsResponse, isLoading: itemsLoading } = useQuery(
    ['userItems', 'recent'],
    () => userItemsApi.getItems({ 
      sort: 'createdAt', 
      order: 'desc', 
      limit: 4 
    }),
    {
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  );

  const stats = statsResponse?.data || {};
  const recentItems = recentItemsResponse?.data || [];

  const getRarityInfo = (rarity) => {
    return RARITY_LEVELS.find(r => r.value === rarity) || RARITY_LEVELS[0];
  };

  const getCategoryIcon = (category) => {
    const categoryIcons = {
      adventure_gear: '🎒',
      collectible: '💎',
      badge: '🏆',
      achievement: '🥇',
      souvenir: '🎁',
      memory: '💭',
      photo: '📸',
      ticket: '🎫',
      certificate: '📜',
      virtual_item: '💻',
      reward: '🎖️',
      token: '🪙',
      other: '📦'
    };
    return categoryIcons[category] || '📦';
  };

  if (statsLoading && itemsLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-white/20 rounded w-1/3"></div>
          <div className="space-y-3">
            <div className="h-4 bg-white/20 rounded"></div>
            <div className="h-4 bg-white/20 rounded w-2/3"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-6 hover:border-white/40 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="text-2xl">🎒</div>
          <div>
            <h3 className="text-xl font-bold text-white">My Inventory</h3>
            <p className="text-gray-400 text-sm">Adventure collection</p>
          </div>
        </div>
        <Link
          to="/inventory"
          className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-colors text-gray-400 hover:text-white"
          title="View all items"
        >
          <ArrowRightIcon className="w-5 h-5" />
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-white">{stats.totalItems || 0}</div>
          <div className="text-xs text-gray-400">Items</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-yellow-400">{stats.totalValue || 0}</div>
          <div className="text-xs text-gray-400">Points</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-400">{stats.favoriteCount || 0}</div>
          <div className="text-xs text-gray-400">Favorites</div>
        </div>
      </div>

      {/* Recent Items */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-semibold text-gray-300">Recent Items</h4>
          {recentItems.length > 0 && (
            <Link
              to="/inventory"
              className="text-xs text-purple-400 hover:text-purple-300 transition-colors"
            >
              View all
            </Link>
          )}
        </div>

        {recentItems.length > 0 ? (
          <div className="space-y-2">
            {recentItems.map((item) => {
              const rarityInfo = getRarityInfo(item.rarity);
              return (
                <div
                  key={item._id}
                  className="flex items-center gap-3 p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors group cursor-pointer"
                >
                  {/* Item Icon */}
                  <div className="text-lg">{getCategoryIcon(item.category)}</div>
                  
                  {/* Item Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-white text-sm font-medium truncate">
                        {item.name}
                      </p>
                      {item.metadata?.isFavorite && (
                        <HeartIcon className="w-3 h-3 text-red-400 fill-current" />
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      <span className={`${rarityInfo.color}`}>
                        <SparklesIcon className="w-3 h-3 inline mr-1" />
                        {rarityInfo.label}
                      </span>
                      {item.value?.points > 0 && (
                        <>
                          <span>•</span>
                          <span className="text-yellow-400">{item.value.points} pts</span>
                        </>
                      )}
                    </div>
                  </div>

                  {/* View Button */}
                  <Link
                    to="/inventory"
                    className="opacity-0 group-hover:opacity-100 p-1 bg-white/10 hover:bg-white/20 rounded transition-all"
                  >
                    <EyeIcon className="w-4 h-4 text-gray-400" />
                  </Link>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-6">
            <div className="text-4xl mb-2">📦</div>
            <p className="text-gray-400 text-sm mb-3">No items yet</p>
            <Link
              to="/inventory"
              className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-colors"
            >
              <PlusIcon className="w-4 h-4" />
              Add First Item
            </Link>
          </div>
        )}
      </div>

      {/* Category Breakdown */}
      {stats.categoryStats && stats.categoryStats.length > 0 && (
        <div className="mt-6 pt-4 border-t border-white/10">
          <h4 className="text-sm font-semibold text-gray-300 mb-3">Top Categories</h4>
          <div className="space-y-2">
            {stats.categoryStats.slice(0, 3).map((category) => (
              <div key={category._id} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span>{getCategoryIcon(category._id)}</span>
                  <span className="text-gray-300 capitalize">
                    {category._id.replace('_', ' ')}
                  </span>
                </div>
                <span className="text-gray-400">{category.count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="mt-6 pt-4 border-t border-white/10">
        <div className="grid grid-cols-2 gap-3">
          <Link
            to="/inventory"
            className="flex items-center justify-center gap-2 px-3 py-2 bg-white/10 hover:bg-white/20 text-white text-sm rounded-lg transition-colors"
          >
            <EyeIcon className="w-4 h-4" />
            View All
          </Link>
          <Link
            to="/inventory"
            className="flex items-center justify-center gap-2 px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-lg transition-colors"
          >
            <PlusIcon className="w-4 h-4" />
            Add Item
          </Link>
        </div>
      </div>
    </div>
  );
};

export default InventoryWidget;
